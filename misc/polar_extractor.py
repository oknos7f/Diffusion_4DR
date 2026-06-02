import os
import re
import shutil
from pathlib import Path
import zipfile
from tqdm import tqdm

def generate_new_filename(internal_path: str) -> str:
    """
    ZIP 내 파일 경로를 파싱하여 고유한 새로운 파일명을 생성합니다.
    """
    internal_path = internal_path.replace('\\', '/')
    match = re.search(r'(?:^|/)(\d+)/polar3d_(\d+)\.npy', internal_path)
    if not match:
        return None
    dir_num = int(match.group(1))
    file_idx = match.group(2)
    return f"{dir_num:02d}{file_idx}.npy"

def process_from_zip(zip_path: str, dst_dir: str):
    """
    ZIP 파일에서 유효한 npy 파일을 추출하여 목적지 디렉토리에 정제된 이름으로 저장합니다.
    기존에 추출된 중복 파일은 스킵하며, tqdm을 통해 처리 프로세스를 실시간으로 표시합니다.
    """
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()
        print(f"[*] ZIP 내 전체 항목 개수: {len(file_list)}개")
        print("[*] ZIP 내부 경로 샘플 (상위 5개):", file_list[:5])
        
        extracted_count = 0
        skipped_count = 0
        
        # 1. 파일 개수 및 진행 프로세스 시각화를 위한 tqdm 래핑
        for name in tqdm(file_list, desc=f"Processing {Path(zip_path).name}", unit="file"):
            new_filename = generate_new_filename(name)
            if not new_filename:
                continue
            
            target_file_path = dst_path / new_filename
            
            # 2. 중복 파일 건너뛰기 검증 (파일이 이미 존재하면 I/O 작업을 생략)
            if target_file_path.exists():
                skipped_count += 1
                continue
            
            with z.open(name) as source, open(target_file_path, 'wb') as target:
                shutil.copyfileobj(source, target)
            extracted_count += 1
            
        print(f"\n[+] 처리 완료: 새롭게 저장된 파일 {extracted_count}개 / 중복으로 건너뛴 파일 {skipped_count}개")

if __name__ == "__main__":
    TARGET_DIR = "./dataset/data/polar3d"
    ZIP_FILE = "./dataset/1-15.zip"
    
    print(f"[*] 현재 작업 디렉토리(CWD): {os.getcwd()}")
    print(f"[*] 타겟 ZIP 절대 경로: {os.path.abspath(ZIP_FILE)}")
    
    if os.path.exists(ZIP_FILE):
        process_from_zip(ZIP_FILE, TARGET_DIR)
    else:
        raise FileNotFoundError(f"지정한 위치에 ZIP 파일이 존재하지 않습니다: {os.path.abspath(ZIP_FILE)}")