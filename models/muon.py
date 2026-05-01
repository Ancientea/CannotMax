import torch
import torch.optim as optim


class Lion(torch.optim.Optimizer):
    """
    Google 提出的 Lion (EvoLved Sign Momentum) 优化器实现。
    显存占用仅为 AdamW 的一半，且常常具有更好的泛化性能。
    注意：Lion 的学习率通常需要设置为 AdamW 的 1/3 到 1/10。
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # 首先执行 Weight Decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]

                # 初始化动量状态
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # 计算 Lion 特有的更新：符号动量
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])

                # 更新 EMA 动量
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


class Muon(torch.optim.Optimizer):
    """
    根据《Muon is Scalable for LLM Training》技术报告实现的 Muon 优化器。
    """

    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.1, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Muon 不支持稀疏梯度计算')

                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']

                # 应用基于 Nesterov 的动量: M_t = \mu * M_{t-1} + \nabla L_t
                buf.mul_(momentum).add_(grad)
                g = grad + momentum * buf

                original_shape = g.shape
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                a, b, c = 3.4445, -4.7750, 2.0315
                X = g / (g.norm() + 1e-8)

                if X.size(0) > X.size(1):
                    for _ in range(ns_steps):
                        A = X.T @ X
                        B = X @ A
                        C = B @ A
                        X = a * X + b * B + c * C
                else:
                    for _ in range(ns_steps):
                        A = X @ X.T
                        B = A @ X
                        C = A @ B
                        X = a * X + b * B + c * C

                X = X.view(original_shape)

                dim0, dim1 = X.size(0), X.size(1) if X.ndim > 1 else 1
                max_dim = max(dim0, dim1)
                scale = 0.2 * (max_dim ** 0.5)

                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(X, alpha=-lr * scale)

        return loss


def get_muon_lion_optimizers(model, muon_lr, lion_lr, weight_decay=0.1, muon_momentum=0.95):
    """
    提供 Muon + Lion 的组合优化器分发策略。
    对模型的所有 >=2D 参数采用 Muon；
    对包含 embedding 字段或一维参数（如 Bias, LayerNorm）使用 Lion。
    Lion 的学习率已经在外层配置中除以了对应比例。
    """
    muon_params = []
    lion_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim >= 2 and 'embed' not in name.lower():
            muon_params.append(p)
        else:
            lion_params.append(p)

    muon_opt = Muon(muon_params, lr=muon_lr, momentum=muon_momentum, weight_decay=weight_decay)

    # Lion 需要稍微大一点的 weight_decay 以防止过拟合，通常建议比 AdamW 大 3-10 倍
    # 这里我们将其设置为常规 weight_decay 的 3 倍
    lion_opt = Lion(lion_params, lr=lion_lr, weight_decay=weight_decay * 3.0)

    return muon_opt, lion_opt
