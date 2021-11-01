import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-test_file_name", type=str, )
parser.add_argument("-gold_file_name", type=str)

args = parser.parse_args()
test_file_path = f"./datasets/{args.test_file_name}"
gold_file_path = f"./ext/results/{args.gold_file_name}"

with open(test_file_path, "r") as file:
    with open(gold_file_path, "w") as gold_out:
        for jsline in file:
            # print(type(jsline))
            data = json.loads(jsline)
            label = data['extractive']
            article = data['article_original']

            for ind in label:
                gold_sent = article[ind]
                gold_out.write("<t>" + gold_sent + "<\t>")

            gold_out.write("\n")