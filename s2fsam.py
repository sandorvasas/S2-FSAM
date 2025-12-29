

class S2FSAM(Optimizer):
    """
        Single-Step implementation of Friendly SAM.
        https://openreview.net/pdf?id=MJgMMqMDu4
        But we do this with FSAM here, not vanilla sam.
    """

    def __init__(self, base_optimizer, beta=0.99, rho=0.05, adaptive=False):
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.state = base_optimizer.state
        self.beta = beta

        get_wdb_run().config.update(
            {
                "sam_type": "S2-F-SAM",
                "sam_rho": rho,
                "sam_adaptive": adaptive,
            }
        )

        for g in self.param_groups:
            g.setdefault("rho", rho)
            g.setdefault("adaptive", adaptive)

    @torch.no_grad()
    def perturb_weights(self):
        grad_norm, noise_grads = self._grad_norm_fsam()
        idx = -1
        for g in self.param_groups:
            scale = g["rho"] / (grad_norm + 1e-12)
            for p in g["params"]:
                if p.grad is None:
                    continue
                idx += 1

                prev_p = self.state[p]["prev_p"] if "prev_p" in self.state[p] else p

                # calculate perturbation epsilone
                e_w = noise_grads[idx] * scale.to(prev_p)

                if g["adaptive"]:
                    e_w *= torch.pow(prev_p, 2)

                p.add_(e_w)
                self.state[p]["e_w"] = e_w


    @torch.no_grad()
    def restore_weights(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue

                self.state[p]["prev_grad"] = torch.clone(p.grad)
                self.state[p]["prev_p"]    = torch.clone(p)

                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])


    @torch.no_grad()
    def step(self, current_training_step, zero_grad=True):
        
        self.restore_weights()

        # update weights using gradient found at "w + epsilon"
        self.base_optimizer.step(current_training_step)
        self.perturb_weights()
        if zero_grad:
            self.base_optimizer.zero_grad(set_to_none=True)


    @torch.no_grad()
    def _grad_norm_fsam(self):
        """
        Just subtract the first moment from the grad then L2 norm on it
        """
        dev = self.param_groups[0]["params"][0].device
        norm_list, noise_grads = [], []
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue

                if "prev_grad" in self.state[p]:
                    prev_grad = self.state[p]["prev_grad"]
                    prev_p    = self.state[p]["prev_p"]
                else:
                    prev_grad = p.grad
                    prev_p    = p

                if "m" not in self.state[p]:
                    self.state[p]["m"] = torch.zeros_like(prev_grad)
                else:
                    self.state[p]["m"].lerp_(prev_grad, 1.0-self.beta)

                noise = prev_grad - self.state[p]["m"]
                noise_grads.append(noise)

                if g["adaptive"]:
                    noise = noise * torch.abs(prev_p)
                
                norm_list.append(noise.norm(p=2).to(dev))
        grad_norm = torch.norm(torch.stack(norm_list), p=2)
        return grad_norm, noise_grads
