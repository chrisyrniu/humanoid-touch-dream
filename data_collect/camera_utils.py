import subprocess

def list_video_devices():
    # Run the v4l2-ctl command to list devices
    result = subprocess.run(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    
    # Parse the output to map camera names to device paths
    device_map = {}
    lines = output.split('\n')
    current_device_name = ""
    
    for line in lines:
        if line.strip() == "":
            continue
        if not line.startswith('\t'):
            # This is a device name line
            current_device_name = line.strip()
        else:
            # This is a device path line
            device_path = line.strip()
            if current_device_name in device_map:
                device_map[current_device_name].append(device_path)
            else:
                device_map[current_device_name] = [device_path]
    
    return device_map

def find_device_path_by_name(device_map, name):
    for device_name, device_paths in device_map.items():
        if name in device_name:
            return device_paths[0]  # Return the first device path found
    return None