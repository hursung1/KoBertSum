import json

test_file_path = "datasets/aihub/test_500_1_신문기사.jsonl"
gold_file_path = "ext/results/result_0830_1447_step_32500.gold"

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