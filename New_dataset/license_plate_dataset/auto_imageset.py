import os

# Define the directory paths
pwd = os.getcwd()
jpeg_path = pwd + '/JPEGImages'
image_sets_path = pwd + '/ImageSets/Main'

# Define the list of tasks and their corresponding file names
tasks = {
    'train': 'train.txt',
    'val': 'val.txt',
    'test': 'test.txt'
}

# Create the ImageSets folder if it doesn't exist
if not os.path.exists(image_sets_path):
    os.makedirs(image_sets_path)

# Loop through each task and create the corresponding text file
for task, file_name in tasks.items():
    file_path = os.path.join(image_sets_path, file_name)
    with open(file_path, 'w') as f:
        # Get the list of image names (without file extension) for this task
        if task == 'train':
            # Use the first 80% of the images for training
            images = sorted(os.listdir(jpeg_path))[:int(0.8 * len(os.listdir(jpeg_path)))]
        elif task == 'val':
            # Use the next 10% of the images for validation
            images = sorted(os.listdir(jpeg_path))[int(0.8 * len(os.listdir(jpeg_path))):int(0.9 * len(os.listdir(jpeg_path)))]
        else:
            # Use the last 10% of the images for testing
            images = sorted(os.listdir(jpeg_path))[int(0.9 * len(os.listdir(jpeg_path))):]

        # Write the list of image names to the text file
        for image in images:
            image_name = os.path.splitext(image)[0]
            f.write('{}\n'.format(image_name))
