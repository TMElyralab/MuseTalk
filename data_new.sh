#!/bin/bash

# Function to extract video and audio sections
extract_sections() {
  input_video=$1
  base_name=$(basename "$input_video" .mp4)
  output_dir=$2
  split=$3
  duration=$(ffmpeg -i "$input_video" 2>&1 | grep Duration | awk '{print $2}' | tr -d ,)
  IFS=: read -r hours minutes seconds <<< "$duration"
  total_seconds=$((10#${hours}*3600 + 10#${minutes}*60 + 10#${seconds%.*}))
  chunk_size=180  # 3 minutes in seconds
  index=0

  mkdir -p "$output_dir"

  while [ $((index * chunk_size)) -lt $total_seconds ]; do
    start_time=$((index * chunk_size))
    section_video="${output_dir}/${base_name}_part${index}.mp4"
    section_audio="${output_dir}/${base_name}_part${index}.mp3"
    
    ffmpeg -i "$input_video" -ss "$start_time" -t "$chunk_size" -c copy "$section_video"
    ffmpeg -i "$input_video" -ss "$start_time" -t "$chunk_size" -q:a 0 -map a "$section_audio"
    
    # Create and update the config.yaml file
    echo "task_0:" > config.yaml
    echo "  video_path: \"$section_video\"" >> config.yaml
    echo "  audio_path: \"$section_audio\"" >> config.yaml

    # Run the Python script with the current config.yaml
    python -m scripts.data --inference_config config.yaml --folder_name "$base_name"
    
    index=$((index + 1))
  done

  # Clean up save folder
  rm -rf $output_dir
}

# Main script
if [ $# -lt 3 ]; then
  echo "Usage: $0 <train/test> <output_directory> <input_videos...>"
  exit 1
fi

split=$1
output_dir=$2
shift 2
input_videos=("$@")

# Initialize JSON array
json_array="["

for input_video in "${input_videos[@]}"; do
  base_name=$(basename "$input_video" .mp4)
  
  # Extract sections and run the Python script for each section
  extract_sections "$input_video" "$output_dir" "$split"

  # Add entry to JSON array
  json_array+="\"../data/images/$base_name\","
done

# Remove trailing comma and close JSON array
json_array="${json_array%,}]"

# Write JSON array to the correct file
if [ "$split" == "train" ]; then
  echo "$json_array" > train.json
elif [ "$split" == "test" ]; then
  echo "$json_array" > test.json
else
  echo "Invalid split: $split. Must be 'train' or 'test'."
  exit 1
fi

echo "Processing complete."
