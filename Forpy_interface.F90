module Forpy_interface

use MOM_grid,                  only : ocean_grid_type
use MOM_verticalGrid,          only : verticalGrid_type
use MOM_domains,               only : pass_var
use MOM_coms,                  only : PE_here,num_PEs
use forpy_mod,                 only : module_py,list,ndarray,object,tuple
use forpy_mod,                 only : err_print
use forpy_mod,                 only : forpy_initialize,get_sys_path,import_py,print_py
use forpy_mod,                 only : ndarray_create,tuple_create,call_py,cast
use forpy_mod,                 only : forpy_finalize

implicit none; private

#include <MOM_memory.h>

public :: forpy_run_python_init,forpy_run_python,forpy_run_python_finalize

!> Control structure for Python interface
type, public :: python_interface ; private
  type(module_py) :: pymodule
  type(list) :: paths
end type

contains

!> Initialize Forpy with specify Python script and directory
subroutine forpy_run_python_init(CS,python_dir,python_file)
    character(len=*),         intent(in)  :: python_dir   !< The directory in which python scripts are found
    character(len=*),         intent(in)  :: python_file  !< The name of the Python script to read
    type(python_interface),   intent(inout) :: CS         !< Python interface object
    ! Local Variables
    integer :: ierror ! return code from python interfaces
    ierror = forpy_initialize()
    write(*,*) "############ Initialize Forpy ############"
    ierror = get_sys_path(CS%paths)
    ierror = CS%paths%append(python_dir)
    ierror = import_py(CS%pymodule,python_file)
    if (ierror/=0) then; call err_print; endif
    ierror = print_py(CS%pymodule)
    if (ierror/=0) then; call err_print; endif
  
end subroutine forpy_run_python_init

!> !> Send variables to a python script and output the results
subroutine forpy_run_python(WH_u, WH_v, Sx, Sy, G, GV, CS, wh_size_in, BT)
    type(ocean_grid_type),         intent(in)  :: G      !< The ocean's grid structure.
    type(verticalGrid_type),       intent(in)  :: GV     !< The ocean's vertical grid structure.
    type(python_interface),        intent(in)  :: CS     !< Python interface object
  ! Local Variables
    integer, intent(in) :: wh_size_in(4)  ! Subdomain size with wide halos for input
    logical, intent(in) :: BT             !< If true, momentum forcing from CNN is barotropic.
    real, dimension(wh_size_in(1):wh_size_in(2),&
                    wh_size_in(3):wh_size_in(4),&
                    SZK_(GV)), &
                                    intent(in) :: WH_u     ! The zonal velocity with a wide halo [L T-1 ~> m s-1].
    real, dimension(wh_size_in(1):wh_size_in(2),&
                    wh_size_in(3):wh_size_in(4),&
                    SZK_(GV)), &
                                    intent(in) :: WH_v     ! The meridional velocity with a wide halo [L T-1 ~> m s-1].
    real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), &
                                    intent(out) :: Sx      ! CNN output Sx
    real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), &
                                    intent(out) :: Sy      ! CNN output Sy
  ! Local Variables for Forpy
    integer :: ierror ! return code from python interfaces
    type(ndarray) :: u_py,v_py,out_arr      !< u and v in the form of numpy array
    type(object)  :: obj                    !< return objects
    type(tuple)   :: args                   !< input arguments for the Python module
    real, dimension(:,:,:,:), pointer :: out_for  !< outputs from Python module
    integer :: current_pe
    integer :: i, j, k
    integer :: is, ie, js, je, nz, nztemp

    current_pe = PE_here()

    is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; nz = GV%ke
    if (BT) then
      nztemp = 1
    else
      nztemp = nz
    endif

  ! Covert input into Forpy Numpy Arrays 
    if (BT) then
        ierror = ndarray_create(u_py, WH_u(:,:,1))
        ierror = ndarray_create(v_py, WH_v(:,:,1))
      else
        ierror = ndarray_create(u_py, WH_u)
        ierror = ndarray_create(v_py, WH_v)
      endif
      if (ierror/=0) then; call err_print; endif
    
      ! Create Python Argument 
      ierror = tuple_create(args,4)
      if (ierror/=0) then; call err_print; endif
      ierror = args%setitem(0,u_py)
      ierror = args%setitem(1,v_py)
      ierror = args%setitem(2,current_pe)
      ierror = args%setitem(3,num_PEs())
      if (ierror/=0) then; call err_print; endif
    
      ! Invoke Python 
      ierror = call_py(obj, CS%pymodule, "MOM6_testNN", args)
      if (ierror/=0) then; call err_print; endif
      ierror = cast(out_arr, obj)
      if (ierror/=0) then; call err_print; endif
      ierror = out_arr%get_data(out_for, order='C')
      if (ierror/=0) then; call err_print; endif
    
      ! Destroy Objects
      call u_py%destroy
      call v_py%destroy
      call out_arr%destroy
      call obj%destroy
      call args%destroy
    
      ! Output (out_for in C order has index (nk,nj,ni))
                      ! in F order has index (ni,nj,nk)
      Sx = 0.0; Sy = 0.0;
      do k=1,nztemp
        do j=js,je ; do i=is,ie
          Sx(i,j,k) = out_for(k,j-js+1,i-is+1,1) ! if order='C'
          ! Sx(i,j,k) = out_for(1,i-is+1,j-js+1,k) ! if order='F'
          Sy(i,j,k) = out_for(k,j-js+1,i-is+1,2)
        enddo ; enddo
      enddo
      call pass_var(Sx, G%domain)
      call pass_var(Sy, G%domain)
  
end subroutine forpy_run_python 

!> Finalize Forpy
subroutine forpy_run_python_finalize(CS)
    type(python_interface), intent(inout) :: CS !< Python interface object
    write(*,*) "############ Finalize Forpy ############"
    call CS%pymodule%destroy
    call CS%paths%destroy
   
    call forpy_finalize
  
end subroutine forpy_run_python_finalize

end module Forpy_interface