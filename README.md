# posterior_neuruloid_patterning_segmentation
![image](https://github.com/user-attachments/assets/57aa9dec-1d6d-4172-b754-7d23ff31b5cf)

Analysing posterior neuruloid patterning using segmentation method to obtain intensity trends.

### Suggested set-up of virtual environment for project
Create virtual environment for CellPose https://www.cellpose.org that works for your device. Other libraries are standard, listed in `requirements.txt`.
```bash
conda activate cellpose
conda config --env --add channels conda-forge
conda install --file requirements.txt
```

### Pipeline
Details on usage of folders and files. Use notebooks to run pipeline (some are optional) rougly in the order listed. More details of the pipeline are in notebook files.

```
neuruloid_2D_segmentation/
├── fiji_macros/ 
        ├── get_dapi_only.ijm # extract dapi channel only for segmentation
        ├── clahe_images.ijm  # correct stitched dapi images in a folder
├── neuruloid_segmentation/ # contains .py files of all the functions used in notebooks
├── notebooks/ # run processing in notebooks
        ├── choose_best_2D_slice.ipynb      # choose most focussed slice from 3D stack
        ├── correcting_stitching.ipynb      # take clahe dapi images (saved from fiji) and combine with other channels
        ├── 2D_segment_24hr.ipynb           # Segment image, save masks and save raw data to "intensity_data" - ensure dapi channels are extracted into a separate "dapi/" folder
        ├── plotting_mean_intensities.ipynb # Process raw dataframe by getting mean of intensities depending on distance from edge and save to "../processed_data"
        ├── fitting_rd_with_tbxt.ipynb      # comparing data from "../processed_data" to rd simulations
        ├── quantifying_segmentation_accuracy.ipynb # calculate sensitivity and precision of segmentation
        ├── unconstrained_neuruloid.ipynb # analyse TBXT clusters from unconstrained neuruloid
        └── import_helper.py # to import scripts from "../neuruloid_segmentation"
```

### Data storage
Git ignores folder 'images/' in this repo. Store images in folder named 'images/' with the following data structure to run the notebook (or however you like just change in notebook):

```
neuruloid_2D_segmentation/
├── ...
├── images/                           # this folder is ignored by git since will be too large
    └── dataset_name/                 # name of dataset (e.g., single_z_slices_24hr)
        └── experiment type/          # data for experiment type (e.g., circle_ctrl)
            └── neuruloid_parameter/  # data for some neuruloid parameters (e.g., ctrl_350)
    │           ├── sample_1.tif      # full tiff images with all channels
                ├── sample_2.tif
                ├── ...
                └── dapi/             # extract only dapi images from the full tiff images for segmentation
                    ├── sample_1.tif
                    ├── sample_2.tif
                    └── ...
                └── masks/            # masks after segmentation
                    ├── sample_1.npy
                    ├── sample_2.npy
                    └── ...
├── intensity_data/                   # this folder contains extracted data for each neuruloid 
    └── dataset_name/                 # name of dataset (e.g., single_z_slices_24hr)
        └── experiment type/          # data for experiment type (e.g., circle_ctrl)
            └── neuruloid_parameter/  # data for some neuruloid parameters (e.g., ctrl_350)
                ├── sample_1.csv      # csv file for extracted data for each nuclei
                ├── sample_2.csv
                └── ...
        └── pixel_size.txt            # dictionary containing pixel sizes 
├── processed_data/                   # this folder contains processed data across multiple organodis and normalised 
    └── dataset_name/                 # name of dataset (e.g., single_z_slices_24hr)
        └── experiment type/          # data for experiment type (e.g., circle_ctrl)
            └── parameter_1.csv       # data for some neuruloid parameters (e.g., ctrl_350.csv)
            ├──  parameter_2.csv  
            └── ...
        └── pixel_size.txt            # dictionary containing pixel sizes 
    └── simulation_intensity.csv      # data from reaction-diffusion simulation for gamma parameters
    ```
