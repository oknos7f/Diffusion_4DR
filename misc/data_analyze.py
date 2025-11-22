import numpy as np
import os
import utils.data_processing as ct

path = '../dataset/data/conditions/'
files = [path + i for i in os.listdir(path)]

# 압축률 결과를 저장할 리스트 생성
compression_results = []

for condition in files:
    print(f"Processing: {condition}")
    
    # 데이터 로드 및 변환 (중복 연산을 막기 위해 변수에 할당)
    # 원본 코드에서는 polar_to_cartesian을 두 번 호출하므로 이를 한 번으로 줄임
    raw_data = np.load(condition, mmap_mode='r')
    cartesian_data = ct.polar_to_cartesian(raw_data, threshold=99)
    
    num_max = cartesian_data.shape[0]
    
    # 복셀화 진행
    voxel_data = ct.voxelize(cartesian_data, agg='max')
    num_min = voxel_data.shape[0]
    
    # 압축률 계산 (분모가 0일 경우 대비)
    if num_max > 0:
        ratio = (num_max - num_min) / num_max * 100
    else:
        ratio = 0.0
    
    print(f"{num_max} -> {num_min} ({ratio:.2f}%)")
    
    # 결과 저장 (파일 이름과 압축률)
    compression_results.append({'name': condition, 'ratio': ratio})

print("-" * 30)

# 결과가 존재할 경우 최고/최저 값 찾기
if compression_results:
    # key=lambda x: x['ratio']를 사용하여 ratio 값을 기준으로 정렬/비교
    best = max(compression_results, key=lambda x: x['ratio'])
    worst = min(compression_results, key=lambda x: x['ratio'])
    
    print(f"최고 압축률: {best['ratio']:.2f}% (파일: {best['name']})")
    print(f"최저 압축률: {worst['ratio']:.2f}% (파일: {worst['name']})")
else:
    print("처리된 파일이 없습니다.")