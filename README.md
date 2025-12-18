# Dual-Focusing Network: Enhancing ALS Point Cloud Semantic Segmentation with Adaptive Neighborhood Refinement
The repo for the paper Dual-Focusing Network: Enhancing ALS Point Cloud Semantic Segmentation with Adaptive Neighborhood Refinement 



---


## Dependencies
- Ubuntu: 18.04 or higher
- PyTorch: 1.9.0 
- CUDA: 11.1 
- To create conda environment, command as follows:

  ```
  bash env_setup.sh pt
  ```

## Dataset preparation
- Download STPLS3D [dataset](https://docs.google.com/forms/d/e/1FAIpQLSeysqIANfUmBuOYipjMPXFk4t8a850mz6L9GISYTKqGnMQ74w/viewform) and generate sub blocks from the large scale point clouds:

     ```
     python ./tool/generate_blocks.py -d /path/to/dataset/
     ```

## Usage
- Semantic point cloud segmantation and evaluation on WMSC.
  - Train

    - Specify the gpu used and the path of dataset in config and then do training:

      ```
      python tool/train_stpls.py
      ```

  - Test

    - Afer training, you can test the checkpoint as follows:

      ```
      python tool/test_stpls.py
      ```
  - Post-processing

    - Merge original input point cloud blocks with predictions for visualization.  

      ```
      python tool/merge_block_vis.py -d /path/to/inputs -p /path/to/predictions -s /path/to/save/ply
      ```
