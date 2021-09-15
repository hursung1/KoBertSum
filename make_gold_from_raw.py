import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-test_file_path", type=str, )
parser.add_argument("-gold_file_path", type=str)

args = parser.parse_args()
# test_file_path = "datasets/aihub/test_1500_from_val.jsonl"
# gold_file_path = "ext/results/result_0906_1603_step_53500.gold"

with open(args.test_file_path, "r") as file:
    with open(args.gold_file_path, "w") as gold_out:
        for jsline in file:
            # print(type(jsline))
            data = json.loads(jsline)
            label = data['extractive']
            article = data['article_original']

            for ind in label:
                gold_sent = article[ind]
                gold_out.write("<t>" + gold_sent + "<\t>")

            gold_out.write("\n")