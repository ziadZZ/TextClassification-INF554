import pandas as pd
from pathlib import Path

def voting(submissions_set, path_to_submissions :  Path = Path("submissions")) :
    # file = open("voting_1.csv", "w")
    # file.write("id,target_feature\n")
    file_path = path_to_submissions / f"{submissions_set[0]}.csv"
    reponse = pd.read_csv(file_path)
    for k in range(1, len(submissions_set)) :
        file_path = path_to_submissions / f"{submissions_set[k]}.csv"
        submission = pd.read_csv(file_path)
        reponse['target_feature'] +=  submission['target_feature']
    threshold = len(submissions_set) // 2
    reponse['target_feature'] = (reponse['target_feature'] >= threshold) * 1
    file = open("voting_8.csv", "w")
    file.write("id,target_feature\n")
    for index, row in reponse .iterrows():
       file.write(f"{row['id']},{row['target_feature']}\n")      
    file.close()
    

submissions_set = ["submission_1", "submission_2", "submission_3", "submission_4", "submission_5", "submission_6", "submission_7", "submission_8", "submission_1"]

voting(submissions_set)