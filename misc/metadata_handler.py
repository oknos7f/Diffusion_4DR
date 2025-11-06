import zipfile
import os


def read_first_line_from_zip_txt(directory_path, target_txt_filename):
    """
    주어진 디렉토리 내의 모든 zip 파일을 순회하며,
    각 zip 파일 내에서 지정된 .txt 파일의 첫 번째 줄을 읽어 출력합니다.

    Args:
        directory_path (str): zip 파일들이 위치한 디렉토리 경로.
        target_txt_filename (str): zip 파일 내에서 찾고자 하는 .txt 파일의 이름 (예: 'data.txt').
    """
    
    print(f"🔍 디렉토리: '{directory_path}'에서 zip 파일들을 찾고 있습니다.")
    print(f"📄 각 zip 파일 내에서 '{target_txt_filename}'의 첫 줄을 읽습니다.\n")
    
    # 디렉토리 내의 모든 파일 및 폴더를 순회
    for filename in os.listdir(directory_path):
        # 전체 파일 경로 생성
        full_path = os.path.join(directory_path, filename)
        
        # 파일이 zip 파일인지 확인 (확장자 및 실제 파일 여부)
        if filename.endswith('.zip') and os.path.isfile(full_path):
            print(filename, end=' ')
            
            try:
                # zip 파일을 열기
                with zipfile.ZipFile(full_path, 'r') as zf:
                    # zip 파일 내에 찾고자 하는 .txt 파일이 있는지 확인
                    if target_txt_filename in zf.namelist():
                        
                        # 해당 .txt 파일을 열고 읽기
                        # zf.open()은 파일과 유사한 객체를 반환하며, 'rt'는 텍스트 모드로 읽고 유니코드를 처리함을 의미
                        with zf.open(target_txt_filename, 'r') as txt_file:
                            # 첫 번째 줄만 읽기
                            # decode('utf-8')는 zip 파일에서 읽어온 byte stream을 문자열로 변환
                            first_line = txt_file.readline().decode('utf-8').strip()
                            
                            print(first_line)
                    
                    else:
                        print(f"   ❌ '{target_txt_filename}' 파일을 찾을 수 없습니다.")
            
            except zipfile.BadZipFile:
                print(f"   ⚠️ 오류: 이 파일은 유효한 zip 파일이 아닙니다.")
            except Exception as e:
                print(f"   ⚠️ 예기치 않은 오류 발생: {e}")
        
        # zip 파일이 아니거나 디렉토리인 경우 건너뛰기
        # else:
        # print(f"--- ⏩ 파일 건너뛰기: {filename} ---")


# --- 사용 예시 ---

# 1. zip 파일들이 있는 디렉토리 경로를 지정하세요.
# (예시: 현재 스크립트가 실행되는 곳의 'data' 폴더)
# 실제 환경에 맞게 이 경로를 수정해야 합니다.
target_directory = '../dataset/metadata'

# 2. zip 파일 내에서 첫 줄을 읽고자 하는 .txt 파일의 이름을 지정하세요.
# (예시: 모든 zip 파일 안에 'log_info.txt'가 있다고 가정)
target_file = 'description.txt'

# 함수 실행
read_first_line_from_zip_txt(target_directory, target_file)