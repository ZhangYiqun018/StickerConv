import torch

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
	return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def accuracy(output, target, padding, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		if output.shape[-1] < maxk:
			print(f"[WARNING] Less than {maxk} predictions available. Using {output.shape[-1]} for topk.")

		maxk = min(maxk, output.shape[-1])

		# Take topk along the last dimension.
		_, pred = output.topk(maxk, -1, True, True)  # (N, T, topk)

		mask = (target != padding).type(target.dtype)
		target_expand = target[..., None].expand_as(pred)
		correct = pred.eq(target_expand)
		correct = correct * mask[..., None].expand_as(correct)

		res = []
		for k in topk:
			correct_k = correct[..., :k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / mask.sum()))
		return res

def contrastive_acc(logits: torch.Tensor, topk=(1,)) -> torch.Tensor:
	"""
	Args:
		logits: (N, N) predictions.
		target: (N, num_correct_answers) labels.
	"""
	assert len(logits.shape) == 2, logits.shape

	target = torch.arange(len(logits), device=logits.device)
 
	return accuracy(logits, target, -1, topk)