# Install
- `pip install -e`
- `pip install -r requirements.txt`
- Download the data file as in the standard readme

# Get Started

### Generate a dataset
- `python scripts/generate_data.py data/raw/foo --num-grasps 1000` (you can do many more grasps, but to try it out, use 1000)

### Cleanup dataset by removing extra data
- `python scripts/clean_data.py data/raw/foo`

### Convert your raw data into dataset
- `python scripts/construct_datasets.py data/raw/foo data/datasets/foo`

### Visualise a sample from your dataset
- `python scripts/vis_sample.py`

### Run inference on a sample (visualises the sample first)
- `python scripts/sim_grasp_standalone.py`


# Train

### TBC