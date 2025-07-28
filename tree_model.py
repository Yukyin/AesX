import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque  # 使用队列实现BFS
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import torch
from emu3.mllm.processing_emu3 import Emu3Processor
from judge_model import JudgeModel
from gen_model import GenModel
import os 
import datetime
import sys

sys.path.append(os.path.abspath('./code'))
if not os.path.exists('generation_result'):
    os.makedirs('generation_result')
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
result_save_path = f"generation_result/{time}"
os.makedirs(result_save_path)

class ImageNode:
    def __init__(self, image, parent=None):
        self.parent = parent
        self.image = image
        self.better = False
        self.comment = None

class ModelIteractiveGeneration():
    def __init__(self):
        self.judge_model = JudgeModel()
        self.gen_model = GenModel()
        self.prompt_list_index = 0
        self.epoch_index = 0
    
    def generate_image(self, step_index, prompt, ratios= ["1:1"]):
        mm_list, outputs_0 = self.gen_model.gen_image(prompt, ratios)
        # print("mm_list",type(mm_list))
        # print("mm_list",mm_list)
        file_name = None
        for idx, im in enumerate(mm_list):
            if not isinstance(im, Image.Image):
                continue
            # print(type(im))
            file_name = f"result{step_index}_{idx}.png"
            file_name = os.path.join(result_save_path, file_name)
            im.save(file_name)
        if file_name is None:
            raise ValueError("No image generated")
        return file_name
    
    def image_comments(self, prompt, image1):
        comment = self.judge_model.image_comments(prompt, image1)
        return comment
    
    def images_comparation(self, prompt, image1, image2):
        image_index = self.judge_model.images_comparation(prompt, image1, image2)
        return image_index
    
    def check_completion(self, problem, steps, interaction_log):
        completion_prompt = (
            f"Problem:\n{problem}\n\n"
            f"Steps completed:\n" + "\n".join(steps) + "\n\n"
            "Based on the steps above, have you fully solved the problem?\n"
            "**Respond with only one word, either 'Yes' or 'No'. Do not include any additional text.**"
        )
        completion = self.evaluation(completion_prompt)
        interaction_log.append({'agent': 'Agent A', 'message': completion})

        # Normalize the response
        response = completion.strip().lower()

        if response == "yes":
            return True
        elif response == "no":
            return False
        else:
            # Handle unexpected responses
            print(f"Unexpected response during completion check: {completion}")
            return False
    
    def iteractive_generation(self, problem, ratios= ["1:1"]):
        steps = []
        step_index = 0
        num_iterations = 10  # Set a maximum number of steps to prevent infinite loops
        interaction_log = []  # Store interactions for inspection
        interaction_log.append("############ problem ############")
        interaction_log.append(problem)
        metric_generation = self.judge_model.metric_generation(problem)
        prompt = (
                    f"You are generating an image:\n{problem}\n\n"
                    f"Please provide the first attempt (Step {step_index}) of your solution."
                )
        image1_file_name = self.generate_image(step_index, prompt, ratios)
        image_comments = self.image_comments(metric_generation, image1_file_name)
        image_node_root = ImageNode(image1_file_name)
        image_node_root.comment = image_comments
        queue = deque()  # 队列实现BFS
        queue.append(image_node_root)  # 将根节点添加到队列中
        iteration_count = 0
        while queue and iteration_count < num_iterations:
            # 从队列中取出一个节点
            current_node = queue.popleft()
            image1_file_name = current_node.image
            image_comments = current_node.comment
            prompt = (
                f"You are generating an image:\n{problem}\n\n"
                f"here is some comment of your generation:\n{image_comments}\n\n"
                # f"Please provide the next step (Step {step_index}) of your generation."
                )
            print("##########################")
            print(f"Step {step_index}:")
            print("**************************")
            print("prompt",prompt)
            mm_list, image_token, image2_file_name = self.generate_image(step_index, prompt, ratios)
            comment_judger = self.judge_image(f"从两张图片中，选出更高质量的图片，判断依据如下:{metric_generation}", image1_file_name, image2_file_name)
            print("**************************")
            print("comment_judger",comment_judger)
            if "1" in comment_judger:
                image_node = ImageNode(image2_file_name)
                image_node.better = True
                image_node.parent = image_node_root
                image_node.comment = self.image_comments(metric_generation, image2_file_name)
                queue.append(image_node)
            interaction_log.append(f"############ Step {step_index} ############")
            interaction_log.append(prompt)
            if self.check_completion(problem, steps, interaction_log):
                break  # Solution is complete
        print("Done")
        with open(os.path.join(result_save_path,"interaction_log.txt"), "w") as f:
            for interaction in interaction_log:
                f.write(interaction + "\n")

if __name__ == "__main__":
    ite_gen = ModelIteractiveGeneration()
    ite_gen.iteractive_generation("在一片秋天的森林中，一位穿着维多利亚风格长裙的女子坐在一块苔藓覆盖的石头上，她的周围散落着红色和金黄色的枫叶。远处有一座古老的石拱桥，桥上长满了藤蔓，桥下的溪流反射出晚霞的余辉。")
    print("Done")

