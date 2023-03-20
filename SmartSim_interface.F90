module SmartSim_interface

use iso_fortran_env,       only : real32, real64

use MOM_coms,                  only : PE_here,num_PEs
use MOM_grid,                  only : ocean_grid_type
use MOM_error_handler,         only : MOM_error, FATAL,NOTE, is_root_pe
use MOM_cpu_clock,             only : cpu_clock_id, cpu_clock_begin, cpu_clock_end, CLOCK_ROUTINE
use MOM_database_comms,        only : dbclient_type, dbcomms_CS_type


implicit none; private

public :: smartsim_run_python_init,smartsim_run_python,smartsim_run_python_finalize

!> Control structure for SmartSim Python interface
type, public :: smartsim_python_interface ; private
  type(dbclient_type)   :: client !< Pointer to the database client
  type(dbcomms_CS_type) :: SS_CS  !< Control structure for database client used for Python bridge

  logical :: online_analysis !< If true, post the EKE used in MOM6 at every timestep
  character(len=5) :: model_key  = 'ml'  !< Key where the ML-model is stored
  character(len=5) :: script_key  = 'pys'  !< Key where the Python script (in txt form) is stored
  character(len=7) :: key_suffix !< Suffix appended to every key sent to Redis

  ! Clock ids
  integer :: id_client_init   !< Clock id to time initialization of the client
  integer :: id_set_script    !< Clock id to time setting of the script
  integer :: id_set_model     !< Clock id to time setting of the model
  integer :: id_put_tensor    !< Clock id to time put_tensor routine
  integer :: id_run_script1   !< Clock id to time running of the script 1
  integer :: id_run_model     !< Clock id to time running of the ML model
  integer :: id_run_script2   !< Clock id to time running of the script 2
  integer :: id_unpack_tensor !< Clock id to time retrieval of EKE prediction

end type

contains

!> Initialize SmartSim with specify Python script and directory
subroutine smartsim_run_python_init(CS,python_dir,python_file)
    character(len=*),         intent(in)  :: python_dir    !< The directory in which python scripts are found
    character(len=*),         intent(in)  :: python_file   !< The name of the Python script to read
    type(smartsim_python_interface),   intent(inout) :: CS !< Python interface object
    integer :: db_return_code

    ! Set various clock ids
    CS%id_client_init   = cpu_clock_id('(CNN_SS client init)', grain=CLOCK_ROUTINE)
    CS%id_set_script    = cpu_clock_id('(CNN_SS set script)', grain=CLOCK_ROUTINE)
    CS%id_set_model     = cpu_clock_id('(CNN_SS set model)', grain=CLOCK_ROUTINE)
    CS%id_put_tensor    = cpu_clock_id('(CNN_SS put tensor)', grain=CLOCK_ROUTINE)
    CS%id_run_script1   = cpu_clock_id('(CNN_SS run script 1)', grain=CLOCK_ROUTINE)
    CS%id_run_model     = cpu_clock_id('(CNN_SS run model)', grain=CLOCK_ROUTINE)
    CS%id_run_script2   = cpu_clock_id('(CNN_SS run script 2)', grain=CLOCK_ROUTINE)
    CS%id_unpack_tensor = cpu_clock_id('(CNN_SS unpack tensor )', grain=CLOCK_ROUTINE)

    ! Store pointers in control structure
    write(CS%key_suffix, '(A,I6.6)') '_', PE_here()
    if (.not. CS%client%isinitialized()) then
      call cpu_clock_begin(CS%id_client_init)
      db_return_code = CS%client%initialize(.false.)
      if (CS%client%SR_error_parser(db_return_code)) call MOM_error(FATAL, "Database client failed to initialize")
      call MOM_error(NOTE,"Database Client Initialized")
      call cpu_clock_end(CS%id_client_init)

  ! Set the machine learning model
      if (is_root_pe()) then
       call cpu_clock_begin(CS%id_set_model)
       db_return_code = CS%client%set_model_from_file(CS%model_key, &
       "/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/CNN_GPU.pt", &
                                                   "TORCH", device="GPU")
        ! db_return_code = CS%client%set_model_from_file_multigpu(CS%model_key, &
        ! "/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/CNN_GPU.pt", &
        !                                             "TORCH",first_gpu=0,num_gpus=2)
        if (CS%client%SR_error_parser(db_return_code)) call MOM_error(FATAL, "SmartSim: set_model failed")
        call cpu_clock_end(CS%id_set_model)
        call cpu_clock_begin(CS%id_set_script)
        db_return_code = CS%client%set_script_from_file(CS%script_key, "CPU", &
        "/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/testNN_trace.txt")
        if (CS%client%SR_error_parser(db_return_code)) call MOM_error(FATAL, "SmartSim: set_script failed")
        call cpu_clock_end(CS%id_set_script)
      endif
    endif
  
end subroutine smartsim_run_python_init

!> !> Send variables to a python script and output the results
subroutine smartsim_run_python(in1, out1, CS, TopLayer, CNN_HALO_SIZE)
    type(smartsim_python_interface), intent(in)  :: CS     !< Python interface object
  ! Local Variables
    logical, intent(in) :: TopLayer             !< If true, only top layer is used.
    integer, intent(in) :: CNN_HALO_SIZE        !< Halo size at each side of subdomains
    real, dimension(:,:,:,:), &
                                    intent(in) :: in1     ! input variables.
    real, dimension(:,:,:,:), &
                                    intent(inout) :: out1      ! output variables.
  ! Local Variables for smartsim
    integer :: ierror ! return code from python interfaces
    real(kind=real32), dimension(6,size(in1,2)-CNN_HALO_SIZE*2, &
                                   size(in1,3)-CNN_HALO_SIZE*2, &
                                   size(in1,4)) :: out_for  !< outputs from Python module
    
    integer :: db_return_code = 0
    integer :: current_pe
    integer :: hi, hj ! temporary
    integer :: i, j, k, l
    integer :: nztemp, out_num
    integer :: index_global(4) ! absolute begin and end index in the subdomain

    character(len=255), dimension(1) :: input
    character(len=255), dimension(1) :: model_input
    character(len=255), dimension(1) :: model_output
    character(len=255), dimension(1) :: output

    CHARACTER(LEN=80)::FILE_NAME='SS'
    CHARACTER(LEN=80)::TMP_NAME=' '

    current_pe = PE_here()

    if (TopLayer) then
      nztemp = 1
    else
      nztemp = size(in1,4)
    endif

  ! Put arrays into the database  
    call cpu_clock_begin(CS%id_put_tensor)
    if (TopLayer) then
      db_return_code = CS%client%put_tensor("input"//CS%key_suffix, &
      in1(:,:,:,1), shape(in1(:,:,:,1))) + db_return_code
    else
      db_return_code = CS%client%put_tensor("input"//CS%key_suffix, in1, shape(in1)) + db_return_code
    endif
    ! db_return_code = CS%client%put_tensor("PE"//CS%key_suffix, current_pe, [1]) + db_return_code
    ! db_return_code = CS%client%put_tensor("PEs"//CS%key_suffix, num_PEs, [1]) + db_return_code
    ! db_return_code = CS%client%put_tensor("domain_id"//CS%key_suffix, index_global, shape(index_global)) + db_return_code
    if (CS%client%SR_error_parser(db_return_code)) call MOM_error(FATAL, "Putting metadata into the database failed")
    call cpu_clock_end(CS%id_put_tensor)
    
  ! Run torchscript
    input(1) = "input"//CS%key_suffix
    model_input(1) = "model_input"//CS%key_suffix
    model_output(1) = "model_output"//CS%key_suffix
    output(1) = "output"//CS%key_suffix
    ! script 1
    call cpu_clock_begin(CS%id_run_script1)
    db_return_code = CS%client%run_script(CS%script_key, "pre_process", input, model_input)
    if (CS%client%SR_error_parser(db_return_code)) call MOM_error(FATAL, "Run script1 in the database failed")
    call cpu_clock_end(CS%id_run_script1)
    ! ML model
    call cpu_clock_begin(CS%id_run_model)
    db_return_code = CS%client%run_model(CS%model_key, model_input, model_output)
    ! db_return_code = CS%client%run_model_multigpu(CS%model_key, model_input, model_output,offset=PE_here(),first_gpu=0,num_gpus=2)
    if (CS%client%SR_error_parser(db_return_code)) call MOM_error(FATAL, "Run model in the database failed")
    call cpu_clock_end(CS%id_run_model)
    ! script 2
    call cpu_clock_begin(CS%id_run_script2)
    db_return_code = CS%client%run_script(CS%script_key, "post_process", model_output, output)
    if (CS%client%SR_error_parser(db_return_code)) call MOM_error(FATAL, "Run script2 in the database failed")
    call cpu_clock_end(CS%id_run_script2)

  ! extract the output from Python
    call cpu_clock_begin(CS%id_unpack_tensor)
    db_return_code = CS%client%unpack_tensor(output(1), out_for, shape(out_for))
    if (CS%client%SR_error_parser(db_return_code)) call MOM_error(FATAL, "unpack tensor from the database failed")
    call cpu_clock_end(CS%id_unpack_tensor)
    ! write(*,*) "output shape:", size(out_for)

  ! find the margin size (if order='F')
    hi = (size(out1,2) - size(out_for,2))/2
    hj = (size(out1,3) - size(out_for,3))/2
    
  ! Output (out_for in C order has index (nk,nj,ni))
              ! in F order has index (ni,nj,nk)
    out1 = 0.0
    do k=1,nztemp
      do j=1,size(out_for,3) ; do i=1,size(out_for,2); do l=1,size(out_for,1)
        !out1(l,i+hi,j+hj,k) = out_for(k,j,i,l) ! if order='C'
        out1(l,i+hi,j+hj,k) = out_for(l,i,j,k) ! if order='F'
      enddo ; enddo ; enddo
    enddo

    ! if (is_root_pe()) then
    !   TMP_NAME = 'out_for_Sxm_'//TRIM(FILE_NAME)
    !   open(10,file=TMP_NAME)
    !   do j=1,size(out_for,3) 
    !     write(10,100) (out_for(3,i,j,1),i=1,size(out_for,2))
    !   enddo 
    !   close(10)

    !   TMP_NAME = 'out_for_Sym_'//TRIM(FILE_NAME)
    !   open(10,file=TMP_NAME)
    !   do j=1,size(out_for,3) 
    !     write(10,100) (out_for(4,i,j,1),i=1,size(out_for,2))
    !   enddo 
    !   close(10)

    !   TMP_NAME = 'out_for_Sxd_'//TRIM(FILE_NAME)
    !   open(10,file=TMP_NAME)
    !   do j=1,size(out_for,3) 
    !     write(10,100) (out_for(5,i,j,1),i=1,size(out_for,2))
    !   enddo 
    !   close(10)

    !   TMP_NAME = 'out_for_Syd_'//TRIM(FILE_NAME)
    !   open(10,file=TMP_NAME)
    !   do j=1,size(out_for,3) 
    !     write(10,100) (out_for(6,i,j,1),i=1,size(out_for,2))
    !   enddo 
    !   close(10)
    !   100 FORMAT(5000es15.4)
    ! endif
    ! stop'debugging!'
  
end subroutine smartsim_run_python 

!> Finalize smartsim
subroutine smartsim_run_python_finalize(CS)
    type(smartsim_python_interface), intent(inout) :: CS !< Python interface object
  
end subroutine smartsim_run_python_finalize

end module smartsim_interface