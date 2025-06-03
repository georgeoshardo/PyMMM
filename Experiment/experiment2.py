import yaml
import nd2
import dask.array as da
import zarr
import numpy as np
# Import other necessary libraries for processing (skimage, scipy, etc.)
# from skimage.transform import warp
# from pystackreg import StackReg

class PyMMMExperiment:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_path = self.config['data_source']['path']
        
        # Create a mapping from axis name to index and an ordered list of axis names
        self.axis_name_to_index = self.config['axes_map']
        self.axis_index_to_name = {v: k for k, v in self.axis_name_to_index.items() if v is not None}
        
        # Load data
        if self.data_path.lower().endswith('.nd2'):
            # The nd2 library can read metadata about axes.
            # You might want to reconcile this with your YAML or prioritize YAML.
            # For dask=True, nd2.imread returns a dask array.
            self.raw_data = nd2.imread(self.data_path, dask=True)
            
            # Attempt to get axis order and names from nd2 metadata if not fully specified or to validate YAML
            try:
                with nd2.ND2File(self.data_path) as ndfile:
                    self.nd2_metadata = ndfile.metadata
                    self.nd2_experiment_axes = ndfile.experiment # List of Loop objects (e.g., TimeLoop, XYPosLoop)
                    self.nd2_shape = ndfile.shape
                    self.nd2_dtype = ndfile.dtype
                    # You can try to infer axis order and names from nd2_experiment_axes
                    # and compare/merge with your YAML's axes_map.
                    # For now, we'll primarily rely on the YAML for axis interpretation.
            except Exception as e:
                print(f"Could not read detailed nd2 metadata: {e}")
                self.nd2_metadata = None

        elif self.data_path.lower().endswith('.zarr'):
            self.raw_data = da.from_zarr(self.data_path)
        else:
            raise ValueError("Unsupported data_source path. Must be .nd2 or .zarr")

        # Validate dimensions
        num_defined_axes = len([idx for idx in self.axis_name_to_index.values() if idx is not None])
        if self.raw_data.ndim != num_defined_axes:
            raise ValueError(
                f"Number of dimensions in data ({self.raw_data.ndim}) "
                f"does not match number of non-null axes in YAML config ({num_defined_axes}). "
                f"Data shape: {self.raw_data.shape}, YAML axes: {self.axis_name_to_index}"
            )

        self.shape = self.raw_data.shape
        self.dtype = self.raw_data.dtype
        
        # Store actual axis names in order of the data dimensions
        self.ordered_axis_names = [self.axis_index_to_name.get(i, f'dim_{i}') for i in range(self.raw_data.ndim)]

        # Get specific axis indices for convenience
        self.t_axis = self.axis_name_to_index.get('T')
        self.p_axis = self.axis_name_to_index.get('P') # Position/FOV
        self.c_axis = self.axis_name_to_index.get('C')
        self.z_axis = self.axis_name_to_index.get('Z')
        self.y_axis = self.axis_name_to_index.get('Y')
        self.x_axis = self.axis_name_to_index.get('X')

        # Store names for positions, channels if provided
        self.position_names = self.config.get('position_names')
        self.channel_names = self.config.get('channel_names')

        if self.p_axis is not None and not self.position_names:
            self.position_names = [f"P{i:03d}" for i in range(self.shape[self.p_axis])]
        if self.c_axis is not None and not self.channel_names:
            self.channel_names = [f"C{i:02d}" for i in range(self.shape[self.c_axis])]
            
        self.processed_data = self.raw_data # This will be updated by processing steps
        
        # Placeholders for results of processing
        self.registration_reference_images = {} # Keyed by (p_idx, c_idx) or similar
        self.registration_transforms = {}       # Keyed by (t_idx, p_idx)
        self.trench_definitions = {}            # Keyed by p_idx, stores list of (x_lims, y_lims)
