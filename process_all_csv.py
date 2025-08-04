#!/usr/bin/env python3
"""
Batch processor for RaceBox CSV files
Processes all CSV files in the csv/ directory using cmdline.py
"""

import asyncio
import glob
import os
import sys
import time
from pathlib import Path

from cmdline import parse_file


async def process_csv_file(csv_file):
    """Process a single CSV file"""
    print(f"\n{'='*60}")
    print(f"🏁 Processing: {os.path.basename(csv_file)}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = await parse_file(csv_file)
        end_time = time.time()
        duration = end_time - start_time

        print(f"\n✅ SUCCESS: {os.path.basename(csv_file)}")
        print(f"⏱️  Processing time: {duration:.1f} seconds")
        print(f"📁 {result}")
        return True

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"\n❌ FAILED: {os.path.basename(csv_file)}")
        print(f"⏱️  Failed after: {duration:.1f} seconds")
        print(f"💥 Error: {str(e)}")
        return False


async def main():
    """Process all CSV files in the csv directory"""

    # Find all CSV files in the csv directory
    csv_pattern = "csv/*.csv"
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print("❌ No CSV files found in csv/ directory")
        print(f"Looking for pattern: {csv_pattern}")
        return

    print(f"🎬 Found {len(csv_files)} CSV files to process:")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"  {i}. {os.path.basename(csv_file)}")

    print(f"\n🚀 Starting batch processing...")

    # Process files one by one
    total_start = time.time()
    successful = 0
    failed = 0

    for csv_file in csv_files:
        success = await process_csv_file(csv_file)
        if success:
            successful += 1
        else:
            failed += 1

    total_end = time.time()
    total_duration = total_end - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f"🏆 BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total files: {len(csv_files)}")
    print(f"⏱️  Total time: {total_duration:.1f} seconds")
    if successful > 0:
        print(f"📈 Average per file: {total_duration/len(csv_files):.1f} seconds")


if __name__ == "__main__":
    # Check if csv directory exists
    if not os.path.exists("csv"):
        print("❌ csv/ directory not found")
        print("Please create a csv/ directory and place your CSV files there")
        sys.exit(1)

    asyncio.run(main())
