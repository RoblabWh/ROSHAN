from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForQuestionAnswering
import requests
import numpy as np


class LLMPredictor(object):

    def __init__(self):
        self.model_name = "optimum/roberta-base-squad2"
        #self.model_name = "optimum/mistral-1.1b-testing"

        self.model = ORTModelForQuestionAnswering.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

        self.onnx_qa = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)

        self.context = "The quick brown fox jumps over the lazy dog."

    def set_context(self, observation):
        positions = str(observation[4])
        print("Positions: ", positions)
        context = "You are here to evaluate relative positions. You get four positions. These positions are " + positions + "."
        self.context = context

    def predict(self, question):
        pred = self.onnx_qa(question, self.context)
        return pred['answer']


class LLMPredictorCPU(object):

    def __init__(self):
        self.model_name = "bigcode/starcoder"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to_bettertransformer()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)

        self.context = "The quick brown fox jumps over the lazy dog."

    def set_context(self, observation):
        positions = str(observation[4])
        print("Positions: ", positions)
        context = "You are here to evaluate relative positions. You get four positions. These positions are " + positions + "."
        self.context = context

    def predict(self, question):
        pred = self.pipeline(question, self.context)
        return pred['answer']


class LLMPredictorAPI(object):

    def __init__(self, model_id):
        self.API_TOKEN = "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"
        self.API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {self.API_TOKEN}"}
        self.context = "You are ROSHAN-AI, your task is answering user questions and provide information about the system." \
                       "Focus on answering the user question as good as possible, under no circumstances come up with more user questions." \
                       "If the user asks something unrelated answer in capslock and imitate a robot that speaks cryptic information only" \
                       "The system is a fire simulation realized with Cellular Automata approach, inside the simulation there are agents, realized as drones." \
                       "The drones are controlled by a Reinforcement Learning agent, the agent is trained with PPO algorithm." \
                       "The agent is trained to extinguish fires in the simulation." \
                       "I will now provide you with the current states of the agents."
        self.variables = "Currently there are no variables set."

    def set_system_variables(self, observations, rewards):
        positions_str, velocities_str, fire_status_str = self.parse_observations(observations)
        self.variables = f"Agent Variables, each variable consists of multiple timesteps.\n" \
                         f"Positions: {positions_str}" \
                         f"Velocities: {velocities_str}" \
                         f"Fire Status: {fire_status_str}" \
                         f"Reinforcement Learning Variables\n" \
                         f"Rewards: {rewards}"

    def parse_observations(self, observations):
        radius = observations[0][0].shape[1]

        fire_status = observations[1]
        number_of_fires = np.count_nonzero(fire_status[0][0] >0)
        # for fire_stat in fire_status[0]:
        #     number_of_fires.append(np.count_nonzero(fire_stat > 0))
        fire_status_str = f"Fire Status: {number_of_fires}, this represents the number of cells on fire in a {radius}x{radius} field with the agent in the center.\n"

        positions = np.array2string(observations[3][0][0], precision=2, separator=",")
        positions_str = f"Positions: {positions}, represent the relative positions of the drone on the map. NOT the absolute cells\n"

        velocities = np.array2string(observations[2][0][0], precision=2, separator=",")
        velocities_str = f"Velocities: {velocities}, represent the relative velocities of the drone on the map. NOT the absolute velocities\n"
        return positions_str, velocities_str, fire_status_str

    def query(self, payload):
        # Combine messages into a single input string
        # see: https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
        input_text = f"{self.context}\n\n{self.variables}\nUser Question:{payload}\n"
        json_input = {"inputs": input_text, "parameters": {"return_full_text": False, "max_new_tokens": 500}, "options": {"wait_for_model": "true"}}
        response = requests.post(self.API_URL, headers=self.headers, json=json_input)
        return response.json()

    def predict(self, payload):
        data = self.query(payload)
        return data[0]['generated_text']

    def __call__(self, engine, user_input, obs, rewards):
        if engine.AgentIsRunning():
            self.set_system_variables(obs, rewards)
        else:
            self.variables = "Currently there are no variables set."
        answer = self.predict(user_input)
        engine.SendDataToModel(answer)

