### Drawing Animation of Batch Optimization

1. *Optimized trajectory*: Run following command in current directory    
`python fastG2o9.py tf_label_unopt.txt loop_star1.csv noisy.kitti`  

2. *Unoptimized trajectory*:    
	1. Comment optimized code in `fastG2o9.py`. From line `321 to 329`    
	2. Uncomment unoptimized code in file `fastG2o9.py`. From line `332 to 334`    
	3. Run following command in current directory:    
		`python fastG2o9.py tf_label_unopt.txt loop_star1.csv noisy.kitti`