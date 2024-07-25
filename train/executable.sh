#######################################
#### for action generator with

#sbatch run.sh 'flan-t5' 'small' 'google/flan-t5-small'

# sbatch run.sh 'flan-t5' 'base' 'google/flan-t5-base'

sbatch run.sh 'flan-t5' 'large' 'google/flan-t5-large'

# sbatch run.sh 'flan-t5' 'xl' 'google/flan-t5-xl'


# sbatch run.sh 't5' 'small' 't5-small'

# sbatch run.sh 't5' 'base' 't5-base'

# sbatch run.sh 't5' 'large' 't5-large'

# sbatch run.sh 't5' '3b' 't5-3b'


#######################################
#### for sg generator

# sbatch run_sg.sh 'flan-t5' 'small' 'google/flan-t5-small'

# sbatch run_sg.sh 'flan-t5' 'base' 'google/flan-t5-base'

sbatch run_sg.sh 'flan-t5' 'large' 'google/flan-t5-large'

# sbatch run_sg.sh 'flan-t5' 'xl' 'google/flan-t5-xl'


# sbatch run_sg.sh 't5' 'small' 't5-small'

# sbatch run_sg.sh 't5' 'base' 't5-base'

# sbatch run_sg.sh 't5' 'large' 't5-large'

# sbatch run_sg.sh 't5' '3b' 't5-3b'


#######################################
#### for first sg generator

# sbatch run_sg_first.sh 'flan-t5' 'small' 'google/flan-t5-small'

# sbatch run_sg_first.sh 'flan-t5' 'base' 'google/flan-t5-base'

sbatch run_sg_first.sh 'flan-t5' 'large' 'google/flan-t5-large'

# sbatch run_sg_first.sh 'flan-t5' 'xl' 'google/flan-t5-xl'


# sbatch run_sg_first.sh 't5' 'small' 't5-small'

# sbatch run_sg_first.sh 't5' 'base' 't5-base'

# sbatch run_sg_first.sh 't5' 'large' 't5-large'

# sbatch run_sg_first.sh 't5' '3b' 't5-3b'


#######################################
#### for action generator without

# sbatch run_sw.sh 'flan-t5' 'small' 'google/flan-t5-small'

# sbatch run_sw.sh 'flan-t5' 'base' 'google/flan-t5-base'

sbatch run_sw.sh 'flan-t5' 'large' 'google/flan-t5-large'

# sbatch run_sw.sh 'flan-t5' 'xl' 'google/flan-t5-xl'


# sbatch run_sw.sh 't5' 'small' 't5-small'

# sbatch run_sw.sh 't5' 'base' 't5-base'

# sbatch run_sw.sh 't5' 'large' 't5-large'

# sbatch run_sw.sh 't5' '3b' 't5-3b'


