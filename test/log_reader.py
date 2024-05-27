import glob
import time
import os
import re

def find_latest_log_file(pattern):
    log_files = glob.glob(pattern)
    if not log_files:
        return None
    latest_log_file = max(log_files, key=os.path.getctime)
    return latest_log_file

def extract_kpm_report_data(log_line):
    pattern = re.compile(r'dl_bytes=(\d+),.*?dl_prbs=(\d+)')
    match = pattern.search(log_line)
    if match:
        dl_bytes = match.group(1)
        dl_prbs = match.group(2)
        print(f"dl_bytes: {dl_bytes}, dl_prbs: {dl_prbs}")

def follow_log_file(log_file_path):
    with open(log_file_path, 'r') as log_file:
        log_file.seek(0, 2)  # 移动到文件末尾
        try:
            while True:
                line = log_file.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                if "KpmReport" in line:
                    extract_kpm_report_data(line.strip())
        except KeyboardInterrupt:
            print("Stopped following the log file.")

if __name__ == "__main__":
    log_file_pattern = "/var/log/pods/ricxapp_ricxapp-nexran-*/nexran-xapp/0.log"
    
    while True:
        log_file_path = find_latest_log_file(log_file_pattern)
        if log_file_path:
            print(f"Following log file: {log_file_path}")
            follow_log_file(log_file_path)
        else:
            print("No log file found. Retrying...")
            time.sleep(5)
