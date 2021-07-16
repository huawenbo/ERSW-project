class Classic_RNN(Baseline):
	def __init__(self, input_dim, latent_dim, device, 
		concat_mask = False, obsrv_std = 0.1, 
		use_binary_classif = False,
		linear_classifier = False,
		classif_per_tp = False,
		input_space_decay = False,
		cell = "gru", n_units = 100,
		n_labels = 1,
		train_classif_w_reconstr = False):
		super(Classic_RNN, self).__init__(input_dim, latent_dim, device, 
			obsrv_std = obsrv_std, 
			use_binary_classif = use_binary_classif,
			classif_per_tp = classif_per_tp,
			linear_classifier = linear_classifier,
			n_labels = n_labels,
			train_classif_w_reconstr = train_classif_w_reconstr)
		self.concat_mask = concat_mask
		encoder_dim = int(input_dim)
		if concat_mask:
			encoder_dim = encoder_dim * 2
		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, n_units),
			nn.Tanh(),
			nn.Linear(n_units, input_dim),)
		#utils.init_network_weights(self.encoder)
		utils.init_network_weights(self.decoder)
		if cell == "gru":
			self.rnn_cell = GRUCell(encoder_dim + 1, latent_dim) # +1 for delta t
		elif cell == "expdecay":
			self.rnn_cell = GRUCellExpDecay(
				input_size = encoder_dim, 
				input_size_for_decay = input_dim,
				hidden_size = latent_dim, 
				device = device)
		else:
			raise Exception("Unknown RNN cell: {}".format(cell))
		if input_space_decay:
			self.w_input_decay =  Parameter(torch.Tensor(1, int(input_dim))).to(self.device)
			self.b_input_decay =  Parameter(torch.Tensor(1, int(input_dim))).to(self.device)
		self.input_space_decay = input_space_decay
		self.z0_net = lambda hidden_state: hidden_state
	def get_reconstruction(self, time_steps_to_predict, data, truth_time_steps, 
		mask = None, n_traj_samples = 1, mode = None)
		assert(mask is not None)
		n_traj, n_tp, n_dims = data.size()
		if (len(truth_time_steps) != len(time_steps_to_predict)) or (torch.sum(time_steps_to_predict - truth_time_steps) != 0):
			raise Exception("Extrapolation mode not implemented for RNN models")
		# for classic RNN time_steps_to_predict should be the same as  truth_time_steps
		assert(len(truth_time_steps) == len(time_steps_to_predict))
		batch_size = data.size(0)
		zero_delta_t = torch.Tensor([0.]).to(self.device)
		delta_ts = truth_time_steps[1:] - truth_time_steps[:-1]
		delta_ts = torch.cat((delta_ts, zero_delta_t))
		if len(delta_ts.size()) == 1:
			# delta_ts are shared for all trajectories in a batch
			assert(data.size(1) == delta_ts.size(0))
			delta_ts = delta_ts.unsqueeze(-1).repeat((batch_size,1,1))
		input_decay_params = None
		if self.input_space_decay:
			input_decay_params = (self.w_input_decay, self.b_input_decay)
		if mask is not None:
			utils.check_mask(data, mask)
		hidden_state, all_hiddens = run_rnn(data, delta_ts, 
			cell = self.rnn_cell, mask = mask,
			input_decay_params = input_decay_params,
			feed_previous_w_prob = (0. if self.use_binary_classif else 0.5),
			decoder = self.decoder)
		outputs = self.decoder(all_hiddens)
		# Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
		first_point = data[:,0,:]
		outputs = utils.shift_outputs(outputs, first_point)
		extra_info = {"first_point": (hidden_state.unsqueeze(0), 0.0, hidden_state.unsqueeze(0))}
		if self.use_binary_classif:
			if self.classif_per_tp:
				extra_info["label_predictions"] = self.classifier(all_hiddens)
			else:
				extra_info["label_predictions"] = self.classifier(hidden_state).reshape(1,-1)
		# outputs shape: [n_traj_samples, n_traj, n_tp, n_dims]
		return outputs, extra_info
