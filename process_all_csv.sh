#!/bin/bash

# Batch processor for RaceBox CSV files
# Processes all CSV files in the csv/ directory

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if csv directory exists
if [ ! -d "csv" ]; then
    echo -e "${RED}❌ csv/ directory not found${NC}"
    echo "Please create a csv/ directory and place your CSV files there"
    exit 1
fi

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Find all CSV files
csv_files=(csv/*.csv)

# Check if any CSV files exist
if [ ! -e "${csv_files[0]}" ]; then
    echo -e "${RED}❌ No CSV files found in csv/ directory${NC}"
    exit 1
fi

echo -e "${BLUE}🎬 Found ${#csv_files[@]} CSV files to process:${NC}"
for i in "${!csv_files[@]}"; do
    echo -e "  $((i+1)). $(basename "${csv_files[i]}")"
done

echo -e "\n${YELLOW}🚀 Starting batch processing...${NC}"

# Counters
successful=0
failed=0
start_time=$(date +%s)

# Process each file
for csv_file in "${csv_files[@]}"; do
    echo -e "\n${'='*60}"
    echo -e "${BLUE}🏁 Processing: $(basename "$csv_file")${NC}"
    echo -e "${'='*60}"

    file_start=$(date +%s)

    if python cmdline.py "$csv_file"; then
        file_end=$(date +%s)
        duration=$((file_end - file_start))
        echo -e "\n${GREEN}✅ SUCCESS: $(basename "$csv_file")${NC}"
        echo -e "${GREEN}⏱️  Processing time: ${duration} seconds${NC}"
        ((successful++))
    else
        file_end=$(date +%s)
        duration=$((file_end - file_start))
        echo -e "\n${RED}❌ FAILED: $(basename "$csv_file")${NC}"
        echo -e "${RED}⏱️  Failed after: ${duration} seconds${NC}"
        ((failed++))
    fi
done

# Summary
end_time=$(date +%s)
total_duration=$((end_time - start_time))

echo -e "\n${'='*60}"
echo -e "${YELLOW}🏆 BATCH PROCESSING COMPLETE${NC}"
echo -e "${'='*60}"
echo -e "${GREEN}✅ Successful: ${successful}${NC}"
echo -e "${RED}❌ Failed: ${failed}${NC}"
echo -e "${BLUE}📊 Total files: ${#csv_files[@]}${NC}"
echo -e "${BLUE}⏱️  Total time: ${total_duration} seconds${NC}"

if [ $successful -gt 0 ]; then
    avg_time=$((total_duration / ${#csv_files[@]}))
    echo -e "${BLUE}📈 Average per file: ${avg_time} seconds${NC}"
fi
