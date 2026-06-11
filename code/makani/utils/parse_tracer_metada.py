import json

def parse_tracer_metadata(metadata_json_path, params):
    """Helper routine for parsing tracer variable metadata file (e.g., tracer_data.json)."""

    try:
        with open(metadata_json_path, "r") as f:
            metadata = json.load(f)

        # Load basic paths and coordinate grids
        params["tracer_h5_path"] = metadata["h5_path"]
        params["tracer_dhours"] = metadata["dhours"]
        params["tracer_lat"] = metadata["coords"]["lat"]
        params["tracer_lon"] = metadata["coords"]["lon"]
        params["tracer_grid_type"] = metadata["coords"]["grid_type"]

        # Channel name sanitization step for tracer variables
        tracer_channel_names = metadata["coords"]["channel"]
        tracer_channels_idx = []
        if hasattr(params, "tracer_channel_names"):
            for tchn in params["tracer_channel_names"]:
                if tchn not in tracer_channel_names:
                    raise ValueError(f"Error, requested tracer channel {tchn} not found in tracer dataset.")
                else:
                    idx = tracer_channel_names.index(tchn)
                    tracer_channels_idx.append(idx)
        else:
            params["tracer_channel_names"] = tracer_channel_names
            #tracer_channels_idx = list(range(len(tracer_channel_names)))
            tracer_channels_idx = [i for i, name in enumerate(tracer_channel_names) if name.startswith("O2_level_idx_")]

        # Set number of tracer input channels
        params["tracer_in_channels"] = tracer_channels_idx
        params["tracer_out_channels"] = tracer_channels_idx
        params["tracer_input_channels"] = len(tracer_channels_idx)  # <-- Useful for model init
        params["tracer_output_channels"] = len(tracer_channels_idx)

        # Store tracer dataset metadata
        params["tracer_dataset"] = dict(
            name=metadata["dataset_name"],
            description=metadata["attrs"]["description"],
            metadata_file=metadata_json_path
        )

    except Exception as e:
        raise RuntimeError(f"Failed to parse tracer metadata: {str(e)}")

    return params, metadata
