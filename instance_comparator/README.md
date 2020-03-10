# Instance Comparator which detects pairs which are in the same topology(opposite viewpoint). We use a very basic MLP to do this.

## Requirements
You will require numpy(1.14.0), torch(0.4), torchvision(0.2), python 2.7 to run this.

## Inference code
`python run.py`

#### output is stored in out.txt. out.txt contains the true pairs detected by the MLP.

## The visualization of the dataset tested(warehouse-1) can be done by running the following code:
`python mlp_read.py in.txt`
