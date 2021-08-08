# Natale et al. comparison

## Full steps
Steps to run the comparison:
- Place dataset at root of directory
- Generate .pcd files for all .csv files wiht `python python/csv_to_pcd.py`. PCL only uses .pcd files.
- Build C++ code. `cd build`, `cmake ..` `make -j<num processors>`
- Generate SHOT and ESF descriptors. From `scripts/` run `bash compute_all_descriptors.bash`. This uses the c++ built executables.
- Generate the CMESF descriptors. See python/compute_cmesf.py
- Run classifiers - see python/classifier.py

## Compute shot for one pointcloud (.pcd) file
go inside build and run
- cmake ..
- make -jx (x numbers of core to speed up the compilation time)
- /compute_shot ../baseball_0201.pcd ../test ../test .