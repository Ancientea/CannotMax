import logging
from pathlib import Path
import cv2
import numpy as np
import time
import csv
import os
import sys
import recognize
import tqdm

MONSTER_NUM=56
black_list_rows = []

def merge(nums):
    if not nums:
        return ""
    intervals = []
    start = end = nums[0]
    for num in nums[1:]:
        if num == end + 1:
            end = num
        else:
            intervals.append((start, end))
            start = end = num
    intervals.append((start, end))  # 添加最后一个区间
    
    parts = []
    for s, e in intervals:
        if s == e:
            parts.append(str(s))
        else:
            parts.append(f"{s}-{e}")
    return ','.join(parts)

def is_continuous_sublist(sub, main):
    return any(sub == main[i:i+len(sub)] for i in range(len(main) - len(sub) + 1))

def remove_duplicate_subsequences_easy(listdata, threshold=3):
    record = []
    for i in range(len(listdata)-threshold-1):
        if is_continuous_sublist(listdata[i+1:i+threshold+2], listdata[:i+threshold+1]):
            record.extend(list(range(i+1,i+threshold+2)))
    reco_last = list(set(record))
    reco_last.sort()
    processed_data = [listdata[j] for j in range(len(listdata)) if j not in reco_last]
    return processed_data, reco_last

def remove_duplicate_subsequences(arr, threshold=3):
    """
    处理二维数组版本，将每行视为独立元素，避免内存溢出
    :param arr: 二维np数组，形状为(N, D)
    :param threshold: 需要删除的连续重复子序列最小长度
    :return: 去重后的数组，被删除的索引列表
    """
    if arr.ndim != 2:
        raise ValueError("输入必须是二维数组")
    
    n = arr.shape[0]
    print(arr[9])
    # 哈希化每行以便快速比较
    dtype = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    hashed_arr = np.ascontiguousarray(arr).view(dtype).flatten()
    
    # 初始化滚动数组和列最大值记录
    prev_row = np.zeros(n, dtype=int)
    max_per_col = np.zeros(n, dtype=int)
    targ = 10
    for i in range(n):
        # 生成当前行比较掩码
        equal_mask = (hashed_arr[i] == hashed_arr)
        
        # 计算当前行DP值
        curr_row = np.zeros(n, dtype=int)
        curr_row[0] = equal_mask[0]  # 处理j=0
        
        if i > 0:
            # 向量化计算j>=1的情况
            curr_row[1:] = np.where(equal_mask[1:], prev_row[:-1] + 1, 0)
        
        # 更新列最大值（只考虑i<=j的情况）
        col_mask = np.arange(n) > i
        max_per_col = np.maximum(max_per_col, curr_row * col_mask)
        
        # 滚动更新
        prev_row = curr_row
        if i*100/n >= targ:
            print("检查重复数据，处理进度: {:.1f}%: ".format(i*100/n))
            targ += 10
    
    # 确定有效列并生成删除索引
    valid_cols = np.where(max_per_col >= threshold)[0]
    to_remove = set()
    
    for j in valid_cols:
        length = max_per_col[j]
        start = max(0, j - length)
        to_remove.update(range(start, j+1))
    
    final_indices = sorted(to_remove)
    return np.delete(arr, final_indices, axis=0), final_indices

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def read_and_remove_zeros(filename,MONSTER_NUM=56):
    '''
    输入数据文件名
    输出去0和空之后的数组，和删去的行号列表
    '''
    data = []
    datafull = []
    row_id = 0
    kong = []
    short = []
    lines_num = MONSTER_NUM*2
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) < lines_num + 1:
                short.append(row_id)
                data.append([0]*lines_num)
                datafull.append([0]*lines_num)
            # 分离数字部分和末尾的字母
            elif isfloat(row[0]) and '' not in row[:lines_num]:
                numbers = list(map(int,map(float, row[:lines_num])))  # 转换为整数列表
                vals = row[lines_num:]
                datafull.append(numbers+vals)
                data.append(numbers)
            else:
                kong.append(row_id)
                data.append([0]*lines_num)
                datafull.append([0]*lines_num)
                #把这一行转化为全0行暂时录入
            row_id += 1
    print('原数据总长度：',len(datafull))
    print('数据长度过短的行：',merge(short))
    print('含有不合法数据的行：',merge(kong))

    np_array = np.array(data)
    # 去除全零行
    all_zeros = []
    for i in range(np_array.shape[0]):
        if np.all(np_array[i][MONSTER_NUM:] == 0) or np.all(np_array[i][:MONSTER_NUM] == 0):
            all_zeros.append(i)
    #np.delete(np_array, all_zeros, axis=0)

    data_new = [datafull[j] for j in range(len(datafull)) if j not in all_zeros]

    all_zeros_idx = [i for i in all_zeros if i not in kong+short]
    print('一侧数据全为0的行：',merge(all_zeros_idx))
    print('筛选后数据总长度：',len(data_new))
    return data_new,all_zeros,len(datafull)

def do_duplicate(listdata):
    if listdata == []:
        return [],[]
    num_data = [list(map(int,map(float, i[:MONSTER_NUM*2]))) for i in listdata]
    np_num_data = np.array(num_data)
    _,remove_list = remove_duplicate_subsequences(np_num_data, threshold=3)
    #print(remove_list)
    result = [listdata[j] for j in range(len(listdata)) if j not in remove_list]
    return result,remove_list

def ori_pos(n,del1,del2):
    remaining_after_first = [i for i in range(n) if i not in del1]
    second_deleted_original = [remaining_after_first[t] for t in del2]
    all_deleted = del1 + second_deleted_original
    all_deleted.sort()
    return all_deleted,second_deleted_original

def view_monster_counts(listdata):
    if listdata == []:
        return True,[],[]
    wrong_counts = []
    num_left = [list(map(int,map(float, i[:MONSTER_NUM]))) for i in listdata]
    num_right = [list(map(int,map(float, i[MONSTER_NUM:MONSTER_NUM*2]))) for i in listdata]
    print(len(num_left[0]),len(num_right[0]))
    black_listed = False
    MONSTER_MIN = 0
    MONSTER_MAX = 100
    MONSTER_LIMIT = {
        0:[range(0,100),'狗',False],
        1:[range(0,50),'红虫',False],
        2:[range(0,30),'大盾',False],
        3:[range(0,30),'大剑',False],
        6:[range(0,6),'庞贝',False],
        8:[range(0,4),'石头人',False],
        27:[range(0,4),'杰斯顿',False],
        28:[[0],'自在',True],
        29:[[0],'狼主',True],
        30:[[0],'雷德',True]  #三大boss全设为0
                     }
    ind = 0
    for i1,i2 in zip(num_left,num_right):
        for x1 in i1:
            if x1 < MONSTER_MIN:
                print(f'{ind}行左侧发现小于0的数据!')
                if ind not in wrong_counts:
                    wrong_counts.append(ind)
            if x1 > MONSTER_MAX:
                print(f'{ind}行左侧发现大于100的数据!')
                if ind not in wrong_counts:
                    wrong_counts.append(ind)
        for x2 in i2:
            if x2 < MONSTER_MIN:
                print(f'{ind}行右侧发现小于0的数据!')
                if ind not in wrong_counts:
                    wrong_counts.append(ind)
            if x2 > MONSTER_MAX:
                print(f'{ind}行右侧发现大于100的数据!')
                if ind not in wrong_counts:
                    wrong_counts.append(ind)
        for j in MONSTER_LIMIT:
            if i1[j] not in MONSTER_LIMIT[j][0]:
                print(f'{ind}行左侧发现{MONSTER_LIMIT[j][1]}，数量：{i1[j]}')
                if ind not in wrong_counts:
                    wrong_counts.append(ind)
                if MONSTER_LIMIT[j][2] and not black_listed:
                    black_listed = True
                    print(f'确认为30人数据，文档加入黑名单。')
            if i2[j] not in MONSTER_LIMIT[j][0]:
                print(f'{ind}行右侧发现{MONSTER_LIMIT[j][1]}，数量：{i2[j]}')
                if ind not in wrong_counts:
                    wrong_counts.append(ind)
                if MONSTER_LIMIT[j][2] and not black_listed:
                    black_listed = True
                    print(f'确认为30人数据，文档加入黑名单。')
        ind += 1
    processed_data = [listdata[j] for j in range(len(listdata)) if j not in wrong_counts]
    mwdata = is_list_true_np(processed_data)
    processed_data2 = [processed_data[j] for j in range(len(processed_data)) if j not in mwdata]
    print(f'怪物信息不符合权重分配的数据行：{mwdata}')
    return black_listed,wrong_counts,mwdata,processed_data2

def del_duplicate_by_time(listdata,delete_no_time = True):
    ind = 0
    no_time = []
    timedata = []
    wrong_time = []
    for i in listdata:
        if len(i) < MONSTER_NUM*2+2 or i[-1] == 'N/A':
            no_time.append(ind)
        ind += 1
    print(merge(no_time),'行：未发现时间戳！')
    data_with_time = [listdata[j] for j in range(len(listdata)) if j not in no_time]
    ind = 0
    for i in data_with_time:
        if i[-1] not in timedata:
            timedata.append(i[-1])
        else:
            wrong_time.append(ind)
            print(f'{timedata.index(i[-1])}行与{ind}行发现同名截图，文件名：{i[-1]}')
        ind += 1
    if not delete_no_time:
        #先找到wrongtime元素的原始位置，再从原列表删除
        remaining_after_first = [i for i in range(len(listdata)) if i not in no_time]
        second_deleted_original = [remaining_after_first[t] for t in wrong_time]
        data_with_time_ok = [listdata[j] for j in range(len(listdata)) if j not in second_deleted_original]
        wrong_time = second_deleted_original
    else:
        data_with_time_ok = [data_with_time[j] for j in range(len(data_with_time)) if j not in wrong_time]
    return data_with_time_ok, no_time, wrong_time

def savecsv(listdata,outputfile):
    # 处理数字转换
    processed = []
    for row in listdata:
        new_row = []
        for item in row:
            if isinstance(item, (int, float)):
                new_row.append(int(item))
            else:
                new_row.append(item)
        processed.append(new_row)

    # 写入CSV文件
    with open(outputfile, 'w', newline='') as f:
        csv.writer(f).writerows(processed)
    print(f'已保存至{outputfile}')
    

def find_csv_files(root_dir):
    csv_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_path = os.path.join(root, file)
                csv_files.append(csv_path)
    return csv_files

def easydata2data(easydata):
    # 测试用函数
    # easydata格式：[[[序号,数量][序号,数量][序号,数量]],[[序号,数量][序号,数量][序号,数量]],结果]，没有的序号和数量留-1，序号是真实序号
    datalist = [0]*MONSTER_NUM*2
    for i in easydata[0]:
        if i[0] > 0:
            datalist[i[0]-1] = i[1]
    for i in easydata[1]:
        if i[0] > 0:
            datalist[i[0]-1+MONSTER_NUM] = i[1]
    datalist.extend(easydata[2])
    return datalist

def find_where_from(easydata,floder_path):
    # easydata格式：[[[序号,数量][序号,数量][序号,数量]],[[序号,数量][序号,数量][序号,数量]],结果]，没有的序号和数量留-1，序号是真实序号
    datalist = [0]*MONSTER_NUM*2
    from_list = []
    for i in easydata[0]:
        if i[0] > 0:
            datalist[i[0]-1] = i[1]
    for i in easydata[1]:
        if i[0] > 0:
            datalist[i[0]-1+MONSTER_NUM] = i[1]
    datalist.extend(easydata[2])
    print(datalist)
    csvlist = find_csv_files(floder_path)
    for c in csvlist:
        print(c)
    for c in csvlist:
        print(f'正在检查：{c}…………………………')
        lines_num = MONSTER_NUM*2
        datafull = []
        row_id = 0
        with open(c, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if isfloat(row[0]) and '' not in row[:lines_num] and len(row) > lines_num:
                    numbers = list(map(int,map(float, row[:lines_num])))  # 转换为整数列表
                    vals = [row[lines_num]]
                    if datalist == numbers+vals:
                        from_list.append([c,row_id+1])
                        print(f'数据来源于：{c}，第{row_id+1}行！')
                row_id += 1
    str_from = ''
    if from_list != []:
        str_from = "\n".join([i[0] + '第：' + str(i[1]) + '行' for i in from_list])
    print(f'可能的数据来源：{str_from}')

def is_distance_not_over_60(a,b,c,d):
    #a, b = interval1
    #c, d = interval2
    # 判断区间是否有交集
    if max(a, c) <= min(b, d):
        return True  # 有交集时距离为0，未超过60
    # 计算不重叠时的间隔
    if b < c:
        distance = c - b  # interval1在左，interval2在右
    else:
        distance = a - d  # interval2在左，interval1在右
    return distance <= 60

def is_list_true_np(fulllist):
    cost_list = [
        [2,0], [2,0.1], [7,0], [7,0], [3,0.1], [10,0], [25,15], [22,0], [25,100], [7,0],
        [5,0.2], [7,0], [2,0], [15,2], [13,0], [12,1.5],
        [6,0.2], [18,3], [15,3], [18,1], [11,0], [10,1], [16,0], [5,0.5], [15,2], [14,0],
        [30,0], [35,100], [-1,-1], [-1,-1], [-1,-1], [6,0.5], [6,0],
        [16,0], [15,5], [11,0], [26,0], [15,0], [4,0.1], [10,0], [21,0], [5,0.2],
        [18,0], [9,1.5], [8,0.5], [16,0], [21,0], [7,0], [36,10], [10,2], [30,15],
        [25,0], [27,0], [32,6], [25,50], [15,5]
        ]
    round_cost_list = [[50,70],[70,90],[90,110],[110,130],[120,160],[140,180],[160,200],[170,230],[190,250],[210,270]]
    
    # Convert to numpy arrays
    cost_arr = np.array(cost_list)
    round_cost_arr = np.array(round_cost_list)
    round_low = round_cost_arr[:, 0]
    round_high = round_cost_arr[:, 1]
    
    # Validity mask for cost entries not equal to [-1, -1]
    valid_mask = np.all(cost_arr != [-1, -1], axis=1)
    
    # Split the input into left and right parts
    fulllist_np = np.array([i[:112] for i in fulllist], dtype=np.float64)
    N = fulllist_np.shape[0]
    left_part = fulllist_np[:, :56]
    right_part = fulllist_np[:, 56:112]
    
    # Calculate valid entries (cost not [-1,-1] and count >0)
    valid_left = valid_mask[np.newaxis, :] & (left_part > 0)
    valid_right = valid_mask[np.newaxis, :] & (right_part > 0)
    
    # Compute mincostL and maxcostL for left
    left_min_terms = (
        (left_part - 1) * cost_arr[np.newaxis, :, 0] +
        ((left_part - 1) * (left_part - 2) * cost_arr[np.newaxis, :, 1]) / 2 +
        0.01
    ) * valid_left
    mincostL = left_min_terms.sum(axis=1)
    
    left_max_terms = (
        (left_part + 1) * cost_arr[np.newaxis, :, 0] +
        (left_part * (left_part + 1) * cost_arr[np.newaxis, :, 1]) / 2 -
        0.01
    ) * valid_left
    maxcostL = left_max_terms.sum(axis=1)
    
    # Compute mincostR and maxcostR for right
    right_min_terms = (
        (right_part - 1) * cost_arr[np.newaxis, :, 0] +
        ((right_part - 1) * (right_part - 2) * cost_arr[np.newaxis, :, 1]) / 2 +
        0.01
    ) * valid_right
    mincostR = right_min_terms.sum(axis=1)
    
    right_max_terms = (
        (right_part + 1) * cost_arr[np.newaxis, :, 0] +
        (right_part * (right_part + 1) * cost_arr[np.newaxis, :, 1]) / 2 -
        0.01
    ) * valid_right
    maxcostR = right_max_terms.sum(axis=1)
    
    # Check overlap with round costs
    left_low = np.maximum(mincostL[:, np.newaxis], round_low)
    left_high = np.minimum(maxcostL[:, np.newaxis], round_high)
    left_cond = left_low <= left_high
    
    right_low = np.maximum(mincostR[:, np.newaxis], round_low)
    right_high = np.minimum(maxcostR[:, np.newaxis], round_high)
    right_cond = right_low <= right_high
    
    both_cond = left_cond & right_cond
    any_round = np.any(both_cond, axis=1)
    
    # Get indices where no round condition is satisfied
    false_indices = np.where(~any_round)[0].tolist()
    return false_indices

def is_list_true(onelist):
    roundlist = []
    #费用和附加费用，写死在代码里吧，不想读文件了。
    cost_list = [[2,0],[2,0.1],[7,0],[7,0],[3,0.1],[10,0],[25,15],[22,0],[25,100],[7,0],[5,0.2],[7,0],[2,0],[15,2],[13,0],[12,1.5],
        [6,0.2],[18,3],[15,3],[18,1],[11,0],[10,1],[16,0],[5,0.5],[15,2],[14,0],[30,0],[35,100],[-1,-1],[-1,-1],[-1,-1],[6,0.5],[6,0],
        [16,0],[15,5],[11,0],[26,0],[15,0],[4,0.1],[10,0],[21,0],[5,0.2],[18,0],[9,1.5],[8,0.5],[16,0],[21,0],[7,0],[36,10],[10,2],[30,15],
        [25,0],[27,0],[32,6],[25,50],[15,5]]
    #
    round_cost_list = [[50,70],[70,90],[90,110],[110,130],[120,160],[140,180],[160,200],[170,230],[190,250],[210,270]]
    left = onelist[:MONSTER_NUM]
    right = onelist[MONSTER_NUM:MONSTER_NUM*2]
    #result = onelist[MONSTER_NUM*2]
    #print(left,right,result)
    mincostL = sum([(left[i]-1)*cost_list[i][0]+(left[i]-1)*(left[i]-2)*cost_list[i][1]/2+0.01 for i in range(len(left)) if (cost_list[i] != [-1,-1]) and (left[i] > 0)])
    maxcostL = sum([(left[i]+1)*cost_list[i][0]+left[i]*(left[i]+1)*cost_list[i][1]/2-0.01 for i in range(len(left)) if (cost_list[i] != [-1,-1]) and (left[i] > 0)])
    mincostR = sum([(right[i]-1)*cost_list[i][0]+(right[i]-1)*(right[i]-2)*cost_list[i][1]/2+0.01 for i in range(len(right)) if (cost_list[i] != [-1,-1]) and (right[i] > 0)])
    maxcostR = sum([(right[i]+1)*cost_list[i][0]+right[i]*(right[i]+1)*cost_list[i][1]/2-0.01 for i in range(len(right)) if (cost_list[i] != [-1,-1]) and (right[i] > 0)])
    print(mincostL,maxcostL,mincostR,maxcostR)
    for i in range(len(round_cost_list)):
        if max(mincostL, round_cost_list[i][0]) <= min(maxcostL, round_cost_list[i][1]) and max(mincostR, round_cost_list[i][0]) <= min(maxcostR, round_cost_list[i][1]):
            roundlist.append(i)     
    if roundlist != []:
        return True
    else:
        print(f'{onelist}is not true！！！')
        return False
    #return is_distance_not_over_60(mincostL,maxcostL,mincostR,maxcostR)

def recognize_review(data,img_floder,matched_threshold = 0.1,ocr_threshold = 0.5):
    print("正在进行识别数据检查")
    print("data行数：",len(data))
    ref_row = [0] * (recognize.MONSTER_COUNT * 2)
    need_delete = [False] * len(data)
    for idx, row in tqdm.tqdm(enumerate(data), total=len(data), desc="Processing rows"):
        ref_row = [0] * (recognize.MONSTER_COUNT * 2)
        try:
            img_name = row[recognize.MONSTER_COUNT * 2 + 1]
            img_path = img_floder / Path(img_name)
            if not img_path.exists():
                print(f"未找到对应的图像： {img_name} ")
                continue
            img = cv2.imread(img_path)
            main_roi = ((0, 0), (img.shape[1], img.shape[0]))
            results = recognize.process_regions(main_roi, img,matched_threshold,ocr_threshold)
            # 处理结果
            for res in results:
                if "error" in res:
                    print(f"识别失败 行号: {idx}, 图片: {img_name}, 错误类型: {res['error']}", file=sys.stderr)
                    break
                if res["matched_id"]:
                    if res["region_id"] < 3:
                        ref_row[res["matched_id"] - 1] = int(res["number"])
                    else:
                        ref_row[res["matched_id"] - 1 + MONSTER_NUM] = int(res["number"])
            else:
                # 检查数据行是否与参考行匹配
                data_row = row[0 : recognize.MONSTER_COUNT * 2]
                if data_row != ref_row:
                    print(f"找到不匹配的数据行： {idx} 行，对应图片文件: {img_name}", file=sys.stderr)
                    print(f"识别结果 : {ref_row}", file=sys.stderr)
                    print(f"文件数据 : {data_row}", file=sys.stderr)
                    need_delete[idx] = True
                else:
                    need_delete[idx] = False
        except Exception as e:
            logging.exception(f"Error processing line {idx}", e)
            need_delete[idx] = True
    newdata = [row for idx, row in enumerate(data) if not need_delete[idx]]
    deleted = [idx for idx, del_flag in enumerate(need_delete) if del_flag]
    return newdata, deleted

#newdata,deleted,ori_len = read_and_remove_zeros('0502.csv',MONSTER_NUM=56)
#_,inc = remove_duplicate_subsequences()
#print('数据例：',newdata[:3])
#result,deleted2 = do_duplicate(newdata)
#print(deleted,deleted2)
#dt = ori_pos(ori_len,deleted,deleted2)
#print(dt)
#print('筛选后数据总长度：',len(result))
#view_monster_counts(newdata)
#del_duplicate_by_time(newdata)

def process_full(filename,do_remove_duplicate_subsequences = False,delete_no_time = True,open_black_list = True,re_recognize_imgs = False,img_floder = '',matched_threshold=0.1, ocr_threshold=0.5):
    wrong_type_list = []
    newdata,deleted0,ori_len = read_and_remove_zeros(filename,MONSTER_NUM=56)
    deleted1 = []
    if do_remove_duplicate_subsequences:
        newdata,deleted1 = do_duplicate(newdata)
    newdata, deleted2, deleted3 = del_duplicate_by_time(newdata,delete_no_time)
    if not delete_no_time:
        deleted2 = []
    black_listed,deleted4,deleted5,newdata = view_monster_counts(newdata)
    deleted6 = []
    if re_recognize_imgs:
        newdata, deleted6 = recognize_review(newdata,img_floder,matched_threshold, ocr_threshold)
    deleted7 = []
    if open_black_list:
        newdata,deleted7 = process_black_list(newdata)

    dl = deleted0
    flag = 0
    for i in [deleted1,deleted2,deleted3,deleted4,deleted5,deleted6,deleted7]:
        if i != []:
            dl,secori = ori_pos(ori_len,dl,i)
            if flag == 0:
                wrong_type_list.append(['不合法的数据：',merge([i + 1 for i in deleted0])])
                wrong_type_list.append(['重复出现的连续数据*：',merge([i + 1 for i in secori])])
            elif flag == 1:
                wrong_type_list.append(['未包含时间轴的数据：',merge([i + 1 for i in secori])])
            elif flag == 2:
                wrong_type_list.append(['时间轴信息重复的数据：',merge([i + 1 for i in secori])])
            elif flag == 3:
                wrong_type_list.append(['怪物信息错误的数据：',merge([i + 1 for i in secori])])
            elif flag == 4:
                wrong_type_list.append(['不符合出怪权重规则的数据：',merge([i + 1 for i in secori])])
            elif flag == 5:
                wrong_type_list.append(['经图片识别错误的数据*：',merge([i + 1 for i in secori])])
            elif flag == 6:
                wrong_type_list.append(['黑名单内数据：',merge([i + 1 for i in secori])])
        flag += 1
    return black_listed,newdata,dl,wrong_type_list


def test1():
    black_listed,newdata,dl,wrong_type_list = process_full('0502processed.csv')
    dllist = [i + 1 for i in dl]
    print(f'删除了{dllist}行的数据')
    for i in wrong_type_list:
        print(i)
    savecsv(newdata,'0502processed2.csv')

def process_floder(flodername,savefilename,lastsavefilename,do_remove_duplicate_subsequences = True,delete_no_time = True,open_black_list = True,re_recognize_imgs = False,img_floder = '',matched_threshold=0.1, ocr_threshold=0.5):
    '''
    输入：
    flodername：需要处理的文件夹名
    savefilename：全部整合保存到的文件名（不进行总去重）
    lastsavefilename：全部整合并去重保存到的最终文件名
    do_remove_duplicate_subsequences：是否清理连续3个以上重复元素的重复序列
    delete_no_time：是否删除没有时间戳的数据行
    '''
    global black_list_rows
    full_data_list = []
    csvlist = find_csv_files(flodername)
    for csv in csvlist:
        print(csv)
    for csv in csvlist:
        print(f'正在处理：{csv}…………………………')
        black_listed,newdata,dl,wrong_type_list = process_full(csv,do_remove_duplicate_subsequences,delete_no_time,open_black_list,re_recognize_imgs,img_floder,matched_threshold, ocr_threshold)
        dllist = [i + 1 for i in dl]
        print(f'删除了{merge(dllist)}行的数据')
        for i in wrong_type_list:
            print(i)
        if not black_listed:
            #未进黑名单则合并至全部数据
            full_data_list += newdata
        else:
            print(f'该数据为30人局数据，自动进入黑名单，不计入总数据！')
            if len(newdata) < 5000:#不是整合数据
                black_list_rows += newdata
    savecsv(full_data_list,savefilename)
    black_listed,newdata,dl,wrong_type_list = process_full(savefilename,do_remove_duplicate_subsequences,delete_no_time,open_black_list,re_recognize_imgs,img_floder,matched_threshold, ocr_threshold)
    #保存后再总处理去重
    dllist = [i + 1 for i in dl]
    print(f'删除了{merge(dllist)}行的数据')
    for i in wrong_type_list:
        print(i)
    savecsv(newdata,lastsavefilename)
    
def process_black_list(full_data):
    #黑名单里所有的数据检测到重复的就删
    global black_list_rows
    delete_rows = []
    ok_data = []
    idx = 0
    for i in full_data:
        if i in black_list_rows:
            delete_rows.append(idx)
        else:
            ok_data.append(i)
        idx += 1
    print(f'黑名单内数据：{merge(delete_rows)}')
    return ok_data,delete_rows
            
    
def process_file(filename,savefilename,do_remove_duplicate_subsequences = True,delete_no_time = True,open_black_list = True,re_recognize_imgs = False,img_floder = '',matched_threshold=0.1, ocr_threshold=0.5):
    '''
    输入：
    filename：需要处理的文件名
    savefilename：处理后保存到的文件名
    do_remove_duplicate_subsequences：是否清理连续3个以上重复元素的重复序列
    delete_no_time：是否删除没有时间戳的数据行
    '''
    black_listed,newdata,dl,wrong_type_list = process_full(filename,do_remove_duplicate_subsequences,delete_no_time,open_black_list,re_recognize_imgs,img_floder,matched_threshold, ocr_threshold)
    #保存后再总处理去重
    dllist = [i + 1 for i in dl]
    print(f'删除了{merge(dllist)}行的数据')
    for i in wrong_type_list:
        print(i)
    savecsv(newdata,savefilename)
        

#process_floder(r'D:\Backup\Downloads\arcdata','arcdata_fullaa.csv','arcdata_full_washed_plus.csv')

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import threading
import queue

class RedirectText(object):
    def __init__(self, text_widget, log_file="processing.log"):
        self.text_widget = text_widget
        self.log_file = log_file
        self.queue = queue.Queue()
        self.root = text_widget.master
        self.lock = threading.Lock()
        
        # 初始化日志文件
        self.setup_logfile()

    def setup_logfile(self):
        try:
            # 使用追加模式打开日志文件
            self.log_fd = open(self.log_file, "a", encoding="utf-8")
        except Exception as e:
            self.log_fd = None
            self.write(f"无法打开日志文件: {str(e)}\n")

    def write(self, message):
        # 写入日志文件（带线程锁）
        with self.lock:
            if self.log_fd:
                try:
                    self.log_fd.write(message)
                    self.log_fd.flush()  # 确保立即写入磁盘
                except Exception as e:
                    self.log_fd = None
                    self.queue.put(f"日志写入失败: {str(e)}\n")
        
        # 写入队列供界面显示
        self.queue.put(message)
        self.root.after(100, self.update_text)

    def update_text(self):
        while not self.queue.empty():
            msg = self.queue.get_nowait()
            self.text_widget.insert(tk.END, msg)
            self.text_widget.see(tk.END)
    
    def flush(self):
        pass

    def close_logfile(self):
        with self.lock:
            if self.log_fd:
                self.log_fd.close()
                self.log_fd = None

class ProcessingThread(threading.Thread):
    def __init__(self, func, args=(), kwargs={}, callback=None):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.callback = callback
        self.daemon = True
        self.exception = None

    def run(self):
        try:
            self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.exception = e
        finally:
            if self.callback:
                self.callback(self.exception)


def create_gui():
    root = tk.Tk()
    root.title("数据搅拌机")
    root.geometry("800x600")

    # 在此处定义关闭事件处理函数（推荐位置）
    def on_close():
        sys.stdout.close_logfile()  # 关闭日志文件
        if messagebox.askokcancel("退出", "确定要退出程序吗？"):  # 添加确认对话框
            root.destroy()  # 销毁窗口

    # 绑定关闭事件处理
    root.protocol("WM_DELETE_WINDOW", on_close)

    # 创建文本输出区域
    output_text = tk.Text(root, wrap=tk.WORD)
    output_text.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
    sys.stdout = RedirectText(output_text, "data_processing.log")

    # 处理文件夹的Frame
    folder_frame = ttk.LabelFrame(root, text="处理文件夹(处理文件夹及其所有子文件夹下的CSV文件，并合并为一个)")
    folder_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

    # 处理文件夹的组件
    ttk.Label(folder_frame, text="选择文件夹:").grid(row=0, column=0, padx=5, sticky="w")
    folder_path = tk.StringVar()
    folder_entry = ttk.Entry(folder_frame, textvariable=folder_path, width=40)
    folder_entry.grid(row=0, column=1, padx=5)
    ttk.Button(folder_frame, text="浏览", command=lambda: folder_path.set(filedialog.askdirectory())).grid(row=0, column=2, padx=5)

    ttk.Label(folder_frame, text="中间保存文件（不进行最终去重）:").grid(row=1, column=0, padx=5, sticky="w")
    interim_save = tk.StringVar()
    ttk.Entry(folder_frame, textvariable=interim_save, width=40).grid(row=1, column=1, padx=5)
    ttk.Button(folder_frame, text="浏览", command=lambda: interim_save.set(filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]))).grid(row=1, column=2, padx=5)

    ttk.Label(folder_frame, text="最终保存文件:").grid(row=2, column=0, padx=5, sticky="w")
    final_save = tk.StringVar()
    ttk.Entry(folder_frame, textvariable=final_save, width=40).grid(row=2, column=1, padx=5)
    ttk.Button(folder_frame, text="浏览", command=lambda: final_save.set(filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]))).grid(row=2, column=2, padx=5)

    # 复选框
    remove_dup = tk.BooleanVar(value=False)
    ttk.Checkbutton(folder_frame, text="不依赖时间戳清理重复子序列（在大数据集会非常慢，通常关闭）", variable=remove_dup).grid(row=3, column=0, columnspan=3, sticky="w")
    
    del_time = tk.BooleanVar(value=True)
    ttk.Checkbutton(folder_frame, text="删除无时间戳数据", variable=del_time).grid(row=4, column=0, columnspan=3, sticky="w")

    open_black = tk.BooleanVar(value=True)
    ttk.Checkbutton(folder_frame, text="将黑名单文件内的所有数据行同时加入黑名单", variable=open_black).grid(row=5, column=0, columnspan=3, sticky="w")
    
    # 修改后的图片识别行（将复选框和阈值输入放在同一行）
    re_recognize_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(folder_frame, text="启用图片二次识别（必须指定图片路径）", variable=re_recognize_var).grid(row=6, column=0, padx=5, sticky="w")

    # 添加匹配阈值设置
    ttk.Label(folder_frame, text="匹配阈值:").grid(row=6, column=1, padx=(20,5), sticky="e")
    matched_threshold_var = tk.DoubleVar(value=0.1)
    ttk.Entry(folder_frame, textvariable=matched_threshold_var, width=6).grid(row=6, column=2, sticky="w")

    # 添加OCR阈值设置
    ttk.Label(folder_frame, text="OCR阈值:").grid(row=6, column=3, padx=(20,5), sticky="e")
    ocr_threshold_var = tk.DoubleVar(value=0.5)
    ttk.Entry(folder_frame, textvariable=ocr_threshold_var, width=6).grid(row=6, column=4, sticky="w")

    # 调整后续行号（原row=6改为row=7开始）
    ttk.Label(folder_frame, text="图片文件夹路径:").grid(row=7, column=0, padx=5, sticky="w")
    img_folder_path = tk.StringVar()
    ttk.Entry(folder_frame, textvariable=img_folder_path, width=40).grid(row=7, column=1, padx=5)
    ttk.Button(folder_frame, text="浏览", command=lambda: img_folder_path.set(filedialog.askdirectory())).grid(row=7, column=2, padx=5)

    # 调整处理文件夹按钮的行号
    

    # 处理文件夹按钮
    folder_button = ttk.Button(folder_frame, text="执行处理")
    folder_button.grid(row=8, column=0, columnspan=5, pady=5)

    # 处理文件的Frame
    file_frame = ttk.LabelFrame(root, text="处理单个文件")
    file_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

    # 处理文件的组件
    ttk.Label(file_frame, text="选择文件:").grid(row=0, column=0, padx=5, sticky="w")
    file_path = tk.StringVar()
    ttk.Entry(file_frame, textvariable=file_path, width=40).grid(row=0, column=1, padx=5)
    ttk.Button(file_frame, text="浏览", command=lambda: file_path.set(filedialog.askopenfilename())).grid(row=0, column=2, padx=5)

    ttk.Label(file_frame, text="保存路径:").grid(row=1, column=0, padx=5, sticky="w")
    save_path = tk.StringVar()
    ttk.Entry(file_frame, textvariable=save_path, width=40).grid(row=1, column=1, padx=5)
    ttk.Button(file_frame, text="浏览", command=lambda: save_path.set(filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]))).grid(row=1, column=2, padx=5)

    # 复选框
    remove_dup_file = tk.BooleanVar(value=False)
    ttk.Checkbutton(file_frame, text="不依赖时间戳清理重复子序列（在大数据集会非常慢，通常关闭）", variable=remove_dup_file).grid(row=2, column=0, columnspan=3, sticky="w")
    
    del_time_file = tk.BooleanVar(value=True)
    ttk.Checkbutton(file_frame, text="删除无时间戳数据", variable=del_time_file).grid(row=3, column=0, columnspan=3, sticky="w")
    
    open_black_file = tk.BooleanVar(value=True)
    ttk.Checkbutton(file_frame, text="将黑名单文件内的所有数据行同时加入黑名单", variable=open_black_file).grid(row=4, column=0, columnspan=3, sticky="w")
    
    # 修改后的图片识别行
    re_recognize_file_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(file_frame, text="启用图片二次识别（必须指定图片路径）", variable=re_recognize_file_var).grid(row=5, column=0, padx=5, sticky="w")

    # 匹配阈值
    ttk.Label(file_frame, text="匹配阈值:").grid(row=5, column=1, padx=(20,5), sticky="e")
    matched_threshold_file_var = tk.DoubleVar(value=0.1)
    ttk.Entry(file_frame, textvariable=matched_threshold_file_var, width=6).grid(row=5, column=2, sticky="w")

    # OCR阈值
    ttk.Label(file_frame, text="OCR阈值:").grid(row=5, column=3, padx=(20,5), sticky="e")
    ocr_threshold_file_var = tk.DoubleVar(value=0.5)
    ttk.Entry(file_frame, textvariable=ocr_threshold_file_var, width=6).grid(row=5, column=4, sticky="w")

    # 调整后续行号
    ttk.Label(file_frame, text="图片文件夹路径:").grid(row=6, column=0, padx=5, sticky="w")
    img_folder_file_path = tk.StringVar()
    ttk.Entry(file_frame, textvariable=img_folder_file_path, width=40).grid(row=6, column=1, padx=5)
    ttk.Button(file_frame, text="浏览", command=lambda: img_folder_file_path.set(filedialog.askdirectory())).grid(row=6, column=2, padx=5)

    # 调整处理文件按钮的行号
    


    # 处理文件按钮
    file_button = ttk.Button(file_frame, text="执行处理")
    file_button.grid(row=7, column=0, columnspan=5, pady=5)

    # 配置网格权重
    root.grid_rowconfigure(3, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # 按钮回调函数
    def process_folder_wrapper():
        folder = folder_path.get()
        interim = interim_save.get()
        final = final_save.get()
        if not folder or not interim or not final:
            messagebox.showerror("错误", "请填写所有路径")
            return

        folder_button.config(state=tk.DISABLED)
        def callback(e):
            folder_button.config(state=tk.NORMAL)
            if e:
                messagebox.showerror("错误", str(e))
            else:
                messagebox.showinfo("完成", "文件夹处理完成")

        thread = ProcessingThread(
            func=process_floder,
            args=(folder, interim, final, remove_dup.get(), del_time.get(), open_black.get(),re_recognize_var.get(), Path(img_folder_path.get()),matched_threshold_var.get(), ocr_threshold_var.get()),
            callback=callback
        )
        thread.start()

    def process_file_wrapper():
        input_file = file_path.get()
        output_file = save_path.get()
        if not input_file or not output_file:
            messagebox.showerror("错误", "请填写所有路径")
            return

        file_button.config(state=tk.DISABLED)
        def callback(e):
            file_button.config(state=tk.NORMAL)
            if e:
                messagebox.showerror("错误", str(e))
            else:
                messagebox.showinfo("完成", "文件处理完成")

        thread = ProcessingThread(
            func=process_file,
            args=(input_file, output_file, remove_dup_file.get(), del_time_file.get(), open_black_file.get(),re_recognize_file_var.get(), Path(img_folder_file_path.get()),matched_threshold_file_var.get(), ocr_threshold_file_var.get()),
            callback=callback
        )
        thread.start()

    # 绑定按钮命令
    folder_button.config(command=process_folder_wrapper)
    file_button.config(command=process_file_wrapper)

    return root

if __name__ == "__main__":
    # 请确保以下函数已经正确导入或定义：
    # process_floder, process_file, find_csv_files, savecsv

    app = create_gui()
    app.mainloop()
