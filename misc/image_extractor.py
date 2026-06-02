import re
import sys
import zipfile
import os
from pathlib import Path

def extract_cam_images(source_dir: str, target_dir: str) -> None:
    """
    Source 디렉터리의 ZIP 파일 내 cam-front 이미지를 규칙에 따라 Target 디렉터리로 추출합니다.
    실행 전 ZIP 파일명 및 내부 구조의 예외 사항을 검증하는 Pre-execution Validation을 수행합니다.
    """
    src_path = Path(source_dir)
    tgt_path = Path(target_dir)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    
    zip_files = list(src_path.glob("*.zip"))
    if not zip_files:
        print("[Data Absence] Source directory에 zip 파일이 존재하지 않습니다.")
        return

    zip_name_pattern = re.compile(r"^(\d+)_cam.*\.zip$")
    img_name_pattern = re.compile(r"^cam-front/cam-front_(\d+)\.png$")
    
    validation_failed = False
    execution_tasks = []
    
    print("=== Phase 1: Pre-execution Validation Scan ===")
    
    for z_path in zip_files:
        name_match = zip_name_pattern.match(z_path.name)
        if not name_match:
            print(f"[Exception] 파일명 규칙 위반 (ID 파싱 불가): {z_path.name}")
            validation_failed = True
            continue
            
        raw_id = name_match.group(1)
        id_str = f"{int(raw_id):02d}"
        
        try:
            with zipfile.ZipFile(z_path, 'r') as zref:
                namelist = zref.namelist()
                valid_internal_files = []
                
                for name in namelist:
                    img_match = img_name_pattern.match(name)
                    if img_match:
                        valid_internal_files.append((name, img_match.group(1)))
                
                if not valid_internal_files:
                    print(f"[Exception] 내부 구조 불일치 ('cam-front/cam-front_*.png' 부재): {z_path.name}")
                    validation_failed = True
                    continue
                
                for internal_path, num_str in valid_internal_files:
                    target_filename = f"{id_str}{num_str}.png"
                    execution_tasks.append({
                        'zip_path': z_path,
                        'internal_path': internal_path,
                        'target_path': tgt_path / target_filename
                    })
                    
        except zipfile.BadZipFile:
            print(f"[Exception] 손상된 ZIP 파일 (Corrupted file): {z_path.name}")
            validation_failed = True

    print("=============================================")
    
    if validation_failed:
        print("\n[Abort] 규칙에 어긋나는 예외가 발견되어 압축 해제를 진행하지 않고 종료합니다.")
        sys.exit(1)
        
    print(f"[Verification Success] 총 {len(execution_tasks)}개의 태스크가 정상 검증되었습니다. Extracion을 시작합니다.\n")
    
    tgt_path.mkdir(parents=True, exist_ok=True)
    current_zip_path = None
    zref = None
    
    try:
        for task in execution_tasks:
            if current_zip_path != task['zip_path']:
                if zref:
                    zref.close()
                current_zip_path = task['zip_path']
                zref = zipfile.ZipFile(current_zip_path, 'r')
            
            data = zref.read(task['internal_path'])
            task['target_path'].write_bytes(data)
            print(f"[Extract] {task['zip_path'].name} -> {task['target_path'].name} (Overwrite Enforced)")
            
    finally:
        if zref:
            zref.close()
            
    print("\n[Success] 모든 데이터의 Extraction 및 정형화 처리가 완료되었습니다.")



if __name__ == "__main__":
    SOURCE_DIR = "./dataset/raw_files"
    TARGET_DIR = "./dataset/images"
    
    print(f"[*] 현재 작업 디렉토리(CWD): {os.getcwd()}")
    
    # 단일 파일 검증이 아닌 디렉토리 스캔 함수 호출
    extract_cam_images(SOURCE_DIR, TARGET_DIR)