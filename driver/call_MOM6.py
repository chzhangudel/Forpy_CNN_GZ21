from smartsim import Experiment
import glob
import itertools
from helpers import parse_mom6_out
import logging
import time
import pandas as pd
import shutil

logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO,
)

def main():
    timing_dfs = []

    db_cpus = [4,8,16,32,64]
    mom6_cpus = [4,8,16,32,64]

    combinations = itertools.product(db_cpus, mom6_cpus)

    for db_cpu, mom6_cpu, in combinations:
        logging.info(f'Starting: db_cpu={db_cpu}\tmom6_cpu={mom6_cpu}')
        start_time = time.time()

        exp = Experiment("MOM6_run_exp", launcher='slurm')
        # create and start an instance of the Orchestrator database
        # create and start an MOM6 experiment
        srun = exp.create_run_settings(
                exe="/scratch/gpfs/aeshao/dev/MOM6-examples/build/ocean_only/MOM6",
                run_command='srun'
                )
        srun.set_nodes(1)
        srun.set_tasks(mom6_cpu)
        # start MOM6
        model = exp.create_model("MOM6_run", srun)
        model.colocate_db_uds(limit_app_cpus=True, db_cpus=db_cpu, debug=True)

        files = glob.glob('/scratch/gpfs/aeshao/dev/MOM6-examples/ocean_only/double_gyre/*')
        model.attach_generator_files(to_copy=files)
        exp.generate(model, overwrite=True)
        exp.start(model, summary=True, block=True)

        end_time = time.time()
        logging.info(f'Finished: db_cpu={db_cpu}\tmom6_cpu={mom6_cpu}\t{end_time-start_time}')

        tmp_df = parse_mom6_out()
        tmp_df['db_cpu'] = db_cpu
        tmp_df['mom6_cpu'] = mom6_cpu

        timing_dfs.append(tmp_df)
        shutil.rmtree('MOM6_run_exp')
        time.sleep(2)

    full_df = pd.concat(timing_dfs, ignore_index=True)
    full_df.to_csv('timings.csv')

if __name__ == "__main__":
    main()
