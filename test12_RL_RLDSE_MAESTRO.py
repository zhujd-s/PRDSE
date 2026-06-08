import torch
import random
import numpy as np
import pdb
import time
import pandas
import os
from multiprocessing import Process, Lock, Manager, Pool
import sys

from config import config_global
sys.path.append("./util/")
from space import dimension_discrete, design_space, create_space_maestro
from actor import actor_policyfunction, get_log_prob, get_log_prob_rnn
from mlp import mlp_policyfunction, rnn_policyfunction
from evaluation_maestro import evaluation_maestro
from config_analyzer import config_self
from timer import timer
from recorder import recorder, writelog
from metric_controller import TransformerMetricController
from context_buffer import SearchContextBuffer

class RLDSE():
	def __init__(self, iindex):
		self.iindex = iindex

		seed = self.iindex * 10000
		torch.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)

		#### step1 assign model
		self.config = config_self(self.iindex)
		self.nnmodel = self.config.nnmodel
		self.constraints = self.config.constraints
		self.goal = self.config.goal
		self.target = self.config.target
		self.baseline = self.config.baseline
		self.baseline_max = self.config.baseline_max
		self.extended_baseline = getattr(self.config, "extended_baseline", dict())
		self.extended_baseline_max = getattr(self.config, "extended_baseline_max", dict())
		self.config.config_check()
		self.pid = os.getpid()

		## initial DSE_action_space
		#self.layer_num = self.config.layer_num
		#self.DSE_action_space = create_space(self.layer_num)

		self.DSE_action_space = create_space_maestro(self.nnmodel, target = self.target)
		##initial evaluation
		self.evaluation = evaluation_maestro(self.iindex, self.nnmodel, self.pid, self.DSE_action_space)

		#define the hyperparameters
		self.PERIOD_BOUND = self.config.period
		self.SAMPLE_PERIOD_BOUND = 1
		self.GEMA = 0.999 #RL parameter, discount ratio
		self.ALPHA = 0.001 #RL parameter, learning step rate
		self.BUFFER_SIZE = 100
		self.BATCH_SIZE = 1
		self.BASE_LINE = 0
		self.ENTROPY_RATIO = 0.1
		self.noise_std = 0.01

		#initial mlp_policyfunction, every action dimension owns a policyfunction
		#TODO:share the weight of first two layer among policyfunction
		action_scale_list = list()
		for dimension in self.DSE_action_space.dimension_box:
			action_scale_list.append(int(dimension.get_scale()))
		self.policy_type = "MLP"
		if(self.policy_type == "MLP"):
			#self.policyfunction = mlp_policyfunction(self.DSE_action_space.get_lenth(), action_scale_list)
			self.policyfunction = mlp_policyfunction(self.DSE_action_space.const_lenth + self.DSE_action_space.dynamic_lenth, action_scale_list)
		elif(self.policy_type == "RNN"):
			self.policyfunction = rnn_policyfunction(input_lenth=1, action_scale_list=action_scale_list)
		
		#print(f"model:{self.policyfunction.state_dict()}")
		#pdb.set_trace()

		##initial e_greedy_policy_function
		self.actor = actor_policyfunction()

		##initial optimizer
		self.policy_optimizer = torch.optim.Adam(
			self.policyfunction.parameters(), 
			lr=self.ALPHA, 
		)
		self.CONTROLLER_ALPHA = 0.0001
		self.CONTROLLER_REG = 0.001
		self.CONTROLLER_GRAD_NORM = 1.0
		self.CONTROLLER_BASELINE_BETA = 0.9
		self.CONTROLLER_IMPROVEMENT_BONUS = 1.5
		self.LAMBDA_EXT_START = 0.1
		self.LAMBDA_EXT_END = 0.9
		self.LAMBDA_EXT_DELTA_SCALE = 0.1

		#### replay buffer, in order to record and reuse high return trace
		#### buffer is consist of trace list
		self.replay_buffer = list()

		#### data vision related
		self.best_objectvalue = 1000
		self.best_objectvalue_list = list()
		self.t = timer()

		self.max_intrinsic_reward = 0
		self.max_extrinsic_reward = 0
		self.max_reward = 0
		self.margin = 1
		self.is_print_log = True
		self.metric_names = [
			"pe_util", "pe_ac_util", "noc_bw_util", "l1_mem_util", "l2_mem_util",
			"throughput_util", "throughput_per_energy_util", "offchip_bw_util"
		]
		self.context_dim = 11
		self.seq_len = 8
		self.stagnation_counter = 0
		self.context_buffer = SearchContextBuffer(seq_len=self.seq_len, context_dim=self.context_dim)
		self.metric_controller = TransformerMetricController(
			context_dim=self.context_dim,
			seq_len=self.seq_len,
			num_metrics=len(self.metric_names),
			default_metric_weights=torch.full((len(self.metric_names),), 1.0 / len(self.metric_names), dtype=torch.float32),
			default_lambda_ext=self.LAMBDA_EXT_START,
		)
		self.controller_optimizer = torch.optim.Adam(
			self.metric_controller.parameters(),
			lr=self.CONTROLLER_ALPHA,
		)
		self.default_metric_weights_tensor = torch.full((len(self.metric_names),), 1.0 / len(self.metric_names), dtype=torch.float32)
		self.default_lambda_ext = self.LAMBDA_EXT_START
		self.controller_objectvalue_baseline = None
		self.log_table = []
		self.log_table.append(["intrinsic_reward_true", "extrinsic_reward_true", "mixed_reward",
							   "pe_util", "pe_ac_util", "noc_bw_util", "l1_mem_util", "l2_mem_util",
							   "throughput_util", "throughput_per_energy_util", "offchip_bw_util",
							   "mean_margin", "lambda_ext",
							   "metric_weight_pe_util", "metric_weight_pe_ac_util", "metric_weight_noc_bw_util",
							   "metric_weight_l1_mem_util", "metric_weight_l2_mem_util",
							   "metric_weight_throughput_util", "metric_weight_throughput_per_energy_util", "metric_weight_offchip_bw_util"])

		# self.log_table.append(["intrinsic_reward_true", "extrinsic_reward_true",
		# 					   "pe_util", "l2_mem_req", "noc_bw_req", "area", "l1_mem_req", "min_margin", "alpha"])

	def _clip_value(self, value, lower=0.0, upper=1.0):
		return float(min(max(value, lower), upper))

	def _safe_ratio(self, numerator, denominator, default=0.0):
		if(abs(denominator) <= 1e-8):
			return float(default)
		return float(numerator / denominator)

	def _score_by_baseline(self, value, baseline_value, reverse=False, upper=2.0):
		if(abs(baseline_value) <= 1e-8):
			return 0.0
		if(reverse):
			score = baseline_value / max(value, 1e-8)
		else:
			score = value / baseline_value
		return float(min(max(score, 0.0), upper))

	def _build_context_vec(self, progress, objectvalue, best_objectvalue, improvement, stagnation, margin, pe_util, pe_ac_util, noc_bw_util, l1_mem_util, l2_mem_util):
		context_vec = np.array([
			self._clip_value(progress),
			self._clip_value(objectvalue, 0.0, 10.0),
			self._clip_value(best_objectvalue, 0.0, 10.0),
			self._clip_value(improvement),
			self._clip_value(stagnation),
			self._clip_value(margin),
			self._clip_value(pe_util),
			self._clip_value(pe_ac_util),
			self._clip_value(noc_bw_util),
			self._clip_value(l1_mem_util),
			self._clip_value(l2_mem_util),
		], dtype=np.float32)
		return context_vec

	def _get_metric_schedule(self, context_vec, enable_grad=False):
		self.context_buffer.append(context_vec)
		context_seq = self.context_buffer.get_sequence_tensor()
		if(enable_grad):
			self.metric_controller.train()
			return self.metric_controller(context_seq)
		self.metric_controller.eval()
		with torch.no_grad():
			return self.metric_controller(context_seq)

	def _get_schedule_lambda_ext(self, period, period_bound):
		clip = lambda value:min(max(value, 0.1), 0.9)
		alpha = clip((1 - period**2 / max(period_bound**2, 1)) * self.margin)
		return float(1 - alpha)

	def train(self):
		self.t.start("all")
		self.context_buffer.reset()
		period_bound = self.SAMPLE_PERIOD_BOUND + self.PERIOD_BOUND
		for period in range(period_bound):
			#print(f"period:{period}", end="\r")
			#here may need a initial function for action_space
			self.DSE_action_space.status_reset()
			rnn_state = None

			#store status, log_prob, reward and return
			status_list, action_list, return_list = list(), list(), list()
			reward_list = list()
			default_metric_weights = self.default_metric_weights_tensor.detach().cpu().numpy().astype(np.float32)
			metrics = None
			objectvalue = float(self.best_objectvalue)
			intrinsic_reward_true = 0.0
			extrinsic_reward_true = 0.0
			intrinsic_reward = 0.0
			extrinsic_reward = 0.0
			final_reward = 0.0
			mean_marign = self.margin
			pe_util, pe_ac_util, noc_bw_util, l1_mem_util, l2_mem_util = 0.0, 0.0, 0.0, 0.0, 0.0
			throughput_util, throughput_per_energy_util, offchip_bw_util = 0.0, 0.0, 0.0
			metric_weights = default_metric_weights.copy()
			lambda_ext = 0.5
			is_best_improved = False

			for step in range(self.DSE_action_space.get_lenth()): 
				#get status from S
				if(self.policy_type == "MLP"):
					current_status = self.DSE_action_space.get_compact_status(step)
				elif(self.policy_type == "RNN"):
					current_status = self.DSE_action_space.get_current_status(step)
				status_list.append(current_status)

				#use policy function to choose action and acquire log_prob
				#action, probs_noise = self.actor.action_choose_with_no_grad(self.policyfunction, self.DSE_action_space, step)
				if(self.policy_type == "MLP"):
					action = self.actor.action_choose_with_no_grad(self.policyfunction, self.DSE_action_space, current_status, step, std=self.noise_std)
				elif(self.policy_type == "RNN"):
					action, rnn_state = self.actor.action_choose_rnn(self.policyfunction, self.DSE_action_space, current_status, step, rnn_state, std=self.noise_std)

				action_list.append(action)

				#take action and get next state S'
				self.DSE_action_space.sample_one_dimension(step, action)

				#### in MC method, we can only sample in last step
				#### and compute reward R
                    
				#TODO:design a good reward function
				if(step < (self.DSE_action_space.get_lenth() - 1)): #delay reward, only in last step the reward will be asigned
					reward = float(0)
				else:
					all_status = self.DSE_action_space.get_status()
					prev_best_objectvalue = self.best_objectvalue
					self.t.start("eva")
					metrics = self.evaluation.evaluate(all_status)
					self.t.end("eva")
					if(metrics != None):
						self.constraints.multi_update(metrics)

						pe_req, pe_ac_req, noc_bw_req, l1_mem_req, l2_mem_req = metrics["cnt_pes"], metrics["pe_ac_req"], metrics["noc_bw_req"], metrics["l1_mem_req"], metrics["l2_mem_req"]
						throughput = metrics["throughput"]
						throughput_per_energy = metrics["throughput_per_energy"]
						offchip_bw_req = metrics["offchip_bw_req"]
						#### utilizations and margins range in [0,1], metric exceeds the threshold be assigned with util=1 and margin=0
						pe_const, pe_ac_const, noc_bw_const, l1_mem_const, l2_mem_const = \
						min(self.constraints.get_threshold("cnt_pes"),self.baseline_max["cnt_pes"]), \
						metrics["cnt_pes"], \
						min(all_status["noc_bw"],self.baseline_max["noc_bw_req"]), \
						min(self.constraints.get_threshold("l1_mem"),self.baseline_max["l1_mem_req"]), \
						min(self.constraints.get_threshold("l2_mem"),self.baseline_max["l2_mem_req"])
						pe_util, pe_ac_util, noc_bw_util, l1_mem_util, l2_mem_util = min(pe_req/pe_const,1), min(pe_ac_req/pe_ac_const,1), min(noc_bw_req/noc_bw_const,1), min(l1_mem_req/l1_mem_const,1), min(l2_mem_req/l2_mem_const,1)
						# Before this change, the new metrics were normalized by running maxima:
						# throughput_util = self._clip_value(self._safe_ratio(throughput, max(self.max_throughput, 1e-8)))
						# throughput_per_energy_util = self._clip_value(self._safe_ratio(throughput_per_energy, max(self.max_throughput_per_energy, 1e-8)))
						# That made their meaning drift during training.
						throughput_baseline = self.extended_baseline.get("throughput", 0.0)
						throughput_per_energy_baseline = self.extended_baseline.get("throughput_per_energy", 0.0)
						offchip_bw_req_baseline = self.extended_baseline.get("offchip_bw_req", 0.0)
						throughput_util = self._score_by_baseline(throughput, throughput_baseline, reverse=False, upper=2.0)
						throughput_per_energy_util = self._score_by_baseline(throughput_per_energy, throughput_per_energy_baseline, reverse=False, upper=2.0)
						offchip_bw_util = self._score_by_baseline(offchip_bw_req, offchip_bw_req_baseline, reverse=True, upper=2.0)
						# pe_util = 1 - pe_util #PE配置率提高面积会增加，感觉面积变小才好
						pe_margin, pe_ac_margin, noc_bw_margin, l1_mem_margin, l2_mem_margin = 1 - pe_util, 1 - pe_ac_util, 1 - noc_bw_util, 1 - l1_mem_util, 1 - l2_mem_util
						#avg_margin = (pe_margin + noc_bw_margin + l1_mem_margin + l2_mem_margin)/4
						#min_margin = min(pe_margin, pe_ac_margin, noc_bw_margin, l1_mem_margin, l2_mem_margin)
						#self.margin = 0.9*self.margin + 0.1*min_margin
						mean_marign = (pe_margin + pe_ac_margin + noc_bw_margin + l1_mem_margin + l2_mem_margin)/5
						self.margin = 0.9*self.margin + 0.1*mean_marign
						
						#### calculate the extrinsic reward
						objectvalue = metrics[self.goal] / self.baseline[self.goal]
						extrinsic_reward_true = 1 / (objectvalue * self.constraints.get_punishment())
						if(extrinsic_reward_true > self.max_extrinsic_reward): 
							self.max_extrinsic_reward = extrinsic_reward_true
						improvement = self._clip_value(self._safe_ratio(max(prev_best_objectvalue - objectvalue, 0.0), max(prev_best_objectvalue, 1e-8)))
						if(objectvalue < prev_best_objectvalue and self.constraints.is_all_meet()):
							self.stagnation_counter = 0
							is_best_improved = True
						else:
							self.stagnation_counter = self.stagnation_counter + 1
						progress = self._safe_ratio(period, max(period_bound - 1, 1), default=1.0)
						stagnation = self._clip_value(self._safe_ratio(self.stagnation_counter, max(self.seq_len, 1), default=1.0))
						context_vec = self._build_context_vec(
							progress=progress,
							objectvalue=objectvalue,
							best_objectvalue=prev_best_objectvalue,
							improvement=improvement,
							stagnation=stagnation,
							margin=self.margin,
							pe_util=pe_util,
							pe_ac_util=pe_ac_util,
							noc_bw_util=noc_bw_util,
							l1_mem_util=l1_mem_util,
							l2_mem_util=l2_mem_util,
						)
						metric_weights_tensor, lambda_ext_tensor = self._get_metric_schedule(context_vec, enable_grad=True)
						# In the lambda-only ablation, intrinsic metric weights were fixed:
						# learned_metric_weights_tensor, lambda_ext_tensor = self._get_metric_schedule(context_vec, enable_grad=True)
						# metric_weights_tensor = self.default_metric_weights_tensor
						learned_lambda_ext_scalar = lambda_ext_tensor.reshape(-1)[0]
						# Before rolling back, lambda_ext was fully learned:
						# lambda_ext_scalar = learned_lambda_ext_scalar
						schedule_lambda_ext = self._get_schedule_lambda_ext(period, period_bound)
						schedule_lambda_ext_tensor = torch.tensor(schedule_lambda_ext, dtype=torch.float32)
						learned_delta_scalar = (learned_lambda_ext_scalar - 0.5) * (2 * self.LAMBDA_EXT_DELTA_SCALE)
						lambda_ext_scalar = torch.clamp(schedule_lambda_ext_tensor + learned_delta_scalar, 0.0, 1.0)

						#### calculate the intrinsic reward
						util_tensor = torch.tensor([
							pe_util, pe_ac_util, noc_bw_util, l1_mem_util, l2_mem_util,
							throughput_util, throughput_per_energy_util, offchip_bw_util
						], dtype=torch.float32)
						log_sum_tensor = torch.sum(metric_weights_tensor * torch.log(util_tensor + 1e-8))
						intrinsic_reward_true_tensor = torch.exp(log_sum_tensor)
						intrinsic_reward_true = float(intrinsic_reward_true_tensor.detach().cpu().item())
						if(intrinsic_reward_true > self.max_intrinsic_reward): 
							self.max_intrinsic_reward = intrinsic_reward_true

						#### calculate the reward
						intrinsic_reward_tensor = intrinsic_reward_true_tensor / max(self.max_intrinsic_reward, 1e-8)
						extrinsic_reward_tensor = torch.tensor(extrinsic_reward_true / max(self.max_extrinsic_reward, 1e-8), dtype=torch.float32)
						final_reward_tensor = 100 * ((1 - lambda_ext_scalar) * intrinsic_reward_tensor + lambda_ext_scalar * extrinsic_reward_tensor)
						final_reward_scalar = float(final_reward_tensor.detach().cpu().item())
						if(self.controller_objectvalue_baseline is None):
							self.controller_objectvalue_baseline = objectvalue
						# Use objectvalue improvement to decide whether the scheduler update is helpful.
						# Before this change, controller_advantage was based on final_reward itself:
						# controller_advantage = max(final_reward_scalar - self.controller_reward_baseline, 0.0)
						controller_advantage = max(
							(self.controller_objectvalue_baseline - objectvalue) / max(self.controller_objectvalue_baseline, 1e-8),
							0.0,
						)
						if(is_best_improved):
							controller_advantage = max(controller_advantage, improvement) * self.CONTROLLER_IMPROVEMENT_BONUS
						controller_advantage_tensor = torch.tensor(controller_advantage, dtype=torch.float32)
						# Keep the update gate tied to objectvalue improvement, but use the original
						# smooth reward surrogate as the gradient carrier for the controller.
						# Before the objective-style ablation, controller_loss was:
						# controller_loss = -controller_advantage_tensor * final_reward_tensor + controller_reg_loss
						controller_reg_loss = self.CONTROLLER_REG * (
							torch.sum((metric_weights_tensor - self.default_metric_weights_tensor)**2) +
							(learned_lambda_ext_scalar - self.default_lambda_ext)**2
						)
						controller_loss = -controller_advantage_tensor * final_reward_tensor + controller_reg_loss
						self.controller_optimizer.zero_grad()
						controller_loss.backward()
						torch.nn.utils.clip_grad_norm_(self.metric_controller.parameters(), self.CONTROLLER_GRAD_NORM)
						self.controller_optimizer.step()
						self.controller_objectvalue_baseline = self.CONTROLLER_BASELINE_BETA * self.controller_objectvalue_baseline + (1 - self.CONTROLLER_BASELINE_BETA) * objectvalue

						metric_weights = metric_weights_tensor.detach().cpu().numpy().astype(np.float32)
						# Before reintroducing schedule-centered delta, the logged lambda_ext came from progress-target mixing:
						# lambda_ext = float(lambda_ext_scalar.detach().cpu().item())
						lambda_ext = float(lambda_ext_scalar.detach().cpu().item())
						intrinsic_reward = float(intrinsic_reward_tensor.detach().cpu().item())
						extrinsic_reward = float(extrinsic_reward_tensor.detach().cpu().item())
						reward = final_reward_scalar
						final_reward = reward

						# print(f"1. intrinsic_reward_true:{intrinsic_reward_true}, extrinsic_reward_true:{extrinsic_reward_true}")
						# print(f"2. intrinsic_reward:{intrinsic_reward}, extrinsic_reward:{extrinsic_reward}, ###reward####:{reward}")
						# print(f"3. util(pe-nocbw-l1mem-l2mem):{pe_util, noc_bw_util, l1_mem_util, l2_mem_util}")
						# print(f"4. min_margin:{min_margin}, alpha:{alpha}\n")
						# print(f"5. resource(pe-nocbw-l1mem-l2mem):{pe_req, noc_bw_req, l1_mem_req, l2_mem_req}\n")
						# print(f"6. constraint(pe-nocbw-l1mem-l2mem):{pe_const, noc_bw_const, l1_mem_const, l2_mem_const}\n")
						if(self.is_print_log):
							log_key_metrics = []
							log_key_metrics.append(intrinsic_reward_true)
							log_key_metrics.append(extrinsic_reward_true)
							log_key_metrics.append(final_reward)
							log_key_metrics.append(pe_util)
							log_key_metrics.append(pe_ac_util)
							log_key_metrics.append(noc_bw_util)
							log_key_metrics.append(l1_mem_util)
							log_key_metrics.append(l2_mem_util)
							log_key_metrics.append(throughput_util)
							log_key_metrics.append(throughput_per_energy_util)
							log_key_metrics.append(offchip_bw_util)
							log_key_metrics.append(mean_marign)
							log_key_metrics.append(lambda_ext)
							log_key_metrics.extend(metric_weights.tolist())
							self.log_table.append(log_key_metrics)

					else:
						self.stagnation_counter = self.stagnation_counter + 1
						progress = self._safe_ratio(period, max(period_bound - 1, 1), default=1.0)
						stagnation = self._clip_value(self._safe_ratio(self.stagnation_counter, max(self.seq_len, 1), default=1.0))
						context_vec = self._build_context_vec(
							progress=progress,
							objectvalue=objectvalue,
							best_objectvalue=prev_best_objectvalue,
							improvement=0.0,
							stagnation=stagnation,
							margin=self.margin,
							pe_util=pe_util,
							pe_ac_util=pe_ac_util,
							noc_bw_util=noc_bw_util,
							l1_mem_util=l1_mem_util,
							l2_mem_util=l2_mem_util,
						)
						metric_weights_tensor, lambda_ext_tensor = self._get_metric_schedule(context_vec, enable_grad=False)
						metric_weights = metric_weights_tensor.detach().cpu().numpy().astype(np.float32)
						learned_lambda_ext_scalar = lambda_ext_tensor.reshape(-1)[0]
						schedule_lambda_ext = self._get_schedule_lambda_ext(period, period_bound)
						schedule_lambda_ext_tensor = torch.tensor(schedule_lambda_ext, dtype=torch.float32)
						learned_delta_scalar = (learned_lambda_ext_scalar - 0.5) * (2 * self.LAMBDA_EXT_DELTA_SCALE)
						lambda_ext = float(torch.clamp(schedule_lambda_ext_tensor + learned_delta_scalar, 0.0, 1.0).detach().cpu().item())
						reward = 0
						final_reward = 0
						if(self.is_print_log):
							log_key_metrics = [intrinsic_reward_true, extrinsic_reward_true, final_reward, pe_util, pe_ac_util, noc_bw_util, l1_mem_util, l2_mem_util, throughput_util, throughput_per_energy_util, offchip_bw_util, mean_marign, lambda_ext]
							log_key_metrics.extend(metric_weights.tolist())
							self.log_table.append(log_key_metrics)

					#### recording
					if(period < self.SAMPLE_PERIOD_BOUND):
						pass
					else:
						if(objectvalue < self.best_objectvalue and self.constraints.is_all_meet()):
							self.best_objectvalue = objectvalue
							print(f"$$$$$period:{period}, iindex:{self.iindex}, best:{self.best_objectvalue}, metrics:{metrics}")
						self.best_objectvalue_list.append(self.best_objectvalue)
					#print(f"period:{period}, this:{objectvalue}, best:{self.best_objectvalue}, metrics:{metrics}, reward:{reward}", end = '\n')
					#print(f"period:{period}, iindex:{self.iindex}, action_list:{action_list[4:8]}, metrics:{metrics}, , baseline:{self.baseline[self.goal]}")
					#print(f"iindex:{self.iindex}, best:{self.best_objectvalue}, metrics:{metrics}")
				reward_list.append(reward)

			# print(reward_list)
			#compute and record return
			return_g = 0
			T = len(reward_list)
			for t in range(T):
				return_g = reward_list[T-1-t] + self.GEMA * return_g
				return_list.append(torch.tensor(return_g).reshape(1))
			return_list.reverse()

			#### record trace into buffer
			temp_reward = lambda sample:(1 - lambda_ext) * sample["intrinsic_reward_true"]/max(self.max_intrinsic_reward, 1e-8) + lambda_ext * sample["extrinsic_reward_true"]/max(self.max_extrinsic_reward, 1e-8)
			# During the Transformer scheduler experiments we selected replay samples by stored final_reward:
			# temp_reward = lambda sample:sample["final_reward"]
			sample = {"intrinsic_reward_true":intrinsic_reward_true, "extrinsic_reward_true":extrinsic_reward_true, "final_reward":final_reward, "return_list":return_list, "status_list":status_list, "action_list":action_list, "obs":self.DSE_action_space.get_obs()}
			if(len(self.replay_buffer) < self.BUFFER_SIZE):
				self.replay_buffer.append(sample)
			else:
				min_sample = min(self.replay_buffer, key = temp_reward)
				sample_reward = temp_reward(sample)
				min_sample_reward = temp_reward(min_sample)
				if(sample_reward > min_sample_reward):
					index = self.replay_buffer.index(min_sample)
					self.replay_buffer[index] = sample
			
			if(period < self.SAMPLE_PERIOD_BOUND):
				pass
			elif(self.replay_buffer):
				#### compute loss and update actor network
				loss = torch.tensor(0)
				for _ in range(self.BATCH_SIZE):
					#### random sample trace from replay buffer
					#sample_selected = random.choice(self.replay_buffer)
					sample_selected = max(self.replay_buffer, key = temp_reward)
					s_return_list = sample_selected["return_list"]
					s_status_list = sample_selected["status_list"]
					s_action_list = sample_selected["action_list"]
					s_obs = sample_selected["obs"]

					#### compute log_prob and entropy
					T = self.DSE_action_space.get_lenth()
					sample_loss = torch.tensor(0)
					pi_noise = torch.tensor(1)
					if(self.policy_type == "MLP"):
						for t in range(T):
							s_entropy, s_log_prob = get_log_prob(self.policyfunction, self.DSE_action_space, s_status_list[t], s_action_list[t], t)
							return_item = -1 * s_log_prob * (s_return_list[t] - self.BASE_LINE)
							entropy_item = -1 * self.ENTROPY_RATIO * s_entropy
							sample_loss = sample_loss + return_item + entropy_item
					elif(self.policy_type == 'RNN'):
						rnn_state_train = None
						for t in range(T):
							s_entropy, s_log_prob, rnn_state_train = get_log_prob_rnn(self.policyfunction, s_obs[t], s_action_list[t], t, rnn_state_train)
							return_item = -1 * s_log_prob * s_return_list[t]
							entropy_item = -1 * self.ENTROPY_RATIO * s_entropy
							sample_loss = sample_loss + return_item + entropy_item
					#### accumulate loss
					sample_loss = sample_loss / T
					loss = loss + sample_loss
				loss = loss / self.BATCH_SIZE

				self.policy_optimizer.zero_grad()
				loss.backward()
				self.policy_optimizer.step()
			else:
				print("no avaiable sample")

			#self.timer.append({"Sampling:":(te_sampling - ts), "Evaluation":(te_evaluation - te_sampling), "Update":(te_update - te_evaluation)})
		#end for-period
		self.t.end("all")
	#end def-train

def run(args):
	iindex, objective_record, timecost_record = args
	print(f"%%%%TEST{iindex} START%%%%")
	DSE = RLDSE(iindex)
	DSE.train()
	timecost_list = DSE.t.get_list("all")
	evacost = DSE.t.get_sum("eva")
	timecost_list.append(evacost)
	
	DSE.best_objectvalue_list.append(iindex)
	timecost_list.append(iindex)
	objective_record.append(DSE.best_objectvalue_list)
	timecost_record.append(timecost_list)

	writelog(DSE.log_table, iindex)

if __name__ == '__main__':
	algoname = "RLDSE_VGG16_1st_trans_newmetrics"
	use_multiprocess = True
	global_config = config_global()
	TEST_BOUND = global_config.TEST_BOUND
	PROCESS_NUM = global_config.PROCESS_NUM
	SCEN_TYPE = global_config.SCEN_TYPE
	SCEN_NUM = global_config.SCEN_NUM
	PASS = global_config.PASS

	args_list = list()
	objective_record = Manager().list()
	timecost_record = Manager().list()

	if(use_multiprocess):
		args_list = list()
		for iindex in range(TEST_BOUND):
			if(iindex in PASS): continue
			args_list.append((iindex, objective_record, timecost_record))
		pool = Pool(PROCESS_NUM)
		pool.map(run, args_list)
		pool.close()
		pool.join()
	else:
		for iindex in range(TEST_BOUND):
			if(iindex in PASS): continue
			run((iindex, objective_record, timecost_record))

	recorder(algoname, global_config, objective_record, timecost_record)
