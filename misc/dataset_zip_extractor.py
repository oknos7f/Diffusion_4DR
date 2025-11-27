import yaml
import os
import zipfile
import re
from tqdm import tqdm


def process_image_archives(source_path, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"폴더 생성 완료: {target_path}")
    
    zip_files = [f for f in os.listdir(source_path) if f.endswith('.zip')]
    zip_files.sort()
    
    print(f"총 {len(zip_files)}개의 ZIP 파일을 처리합니다.")
    
    for zip_name in tqdm(zip_files, desc="전체 진행률", unit="zip"):
        zip_full_path = os.path.join(source_path, zip_name)
        
        try:
            raw_index = zip_name.split('_')[0]
            if not raw_index.isdigit():
                print(f"\n[Skip] 이름 규칙 불일치: {zip_name}")
                continue
            
            file_index_str = raw_index.zfill(2)
            
            with zipfile.ZipFile(zip_full_path, 'r') as zf:
                target_files = [f for f in zf.namelist() if 'cam-front' in f and f.endswith('.png')]
                
                for member in target_files:
                    match = re.search(r'(\d+)\.png$', member)
                    
                    if match:
                        img_number = match.group(1)  # 00001
                        
                        new_filename = f"{file_index_str}{img_number}.png"
                        save_path = os.path.join(target_path, new_filename)
                        
                        with zf.open(member) as source_file, open(save_path, 'wb') as dest_file:
                            dest_file.write(source_file.read())
            os.remove(zip_full_path)
        
        except Exception as e:
            print(f"\n[Error] {zip_name} 처리 중 오류 발생: {e}")
    
    print("\n모든 작업이 완료되었습니다.")


if __name__ == "__main__":
    config = r"../config/config.yaml"
    with open(config, 'r') as file:
        config_data = yaml.safe_load(file)
        
    dataset = config_data["dataset"]
    data_root = dataset["data_root"]
    
    PATH = r"C:\Users\jdmdj\Desktop\Diffusion_4DR\dataset\raw_files"
    IMAGE_PATH = os.path.join(data_root, dataset["image_dir"])
    CONDITION_PATH = os.path.join(data_root, dataset["condition_dir"])
    
    process_image_archives(PATH, IMAGE_PATH, CONDITION_PATH)