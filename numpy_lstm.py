import numpy as np

class NumpyLSTM:
    """
    Minimal single-layer LSTM + linear head, trained with plain-numpy BPTT + Adam.
    Used because torch/tensorflow are unavailable in this offline sandbox.
    Supports 'regression' (MSE) and 'classification' (BCE) via `task`.
    Input:  X of shape (n_samples, seq_len, n_features)
    Output: y of shape (n_samples,)
    """
    def __init__(self, n_features, hidden=16, task='regression', lr=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.h = hidden
        self.n_features = n_features
        self.task = task
        z = hidden + n_features
        scale = 1.0/np.sqrt(z)
        # combined weight matrices for gates: forget, input, cell(candidate), output
        self.Wf = rng.normal(0, scale, (z, hidden)); self.bf = np.zeros(hidden)
        self.Wi = rng.normal(0, scale, (z, hidden)); self.bi = np.zeros(hidden)
        self.Wc = rng.normal(0, scale, (z, hidden)); self.bc = np.zeros(hidden)
        self.Wo = rng.normal(0, scale, (z, hidden)); self.bo = np.zeros(hidden)
        self.Wy = rng.normal(0, 1.0/np.sqrt(hidden), (hidden, 1)); self.by = np.zeros(1)
        self.lr = lr
        self.params = ['Wf','bf','Wi','bi','Wc','bc','Wo','bo','Wy','by']
        self.m = {p: np.zeros_like(getattr(self,p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self,p)) for p in self.params}
        self.t = 0

    @staticmethod
    def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))

    def forward(self, X):
        n, T, f = X.shape
        h = self.h
        Hs = np.zeros((n, T+1, h)); Cs = np.zeros((n, T+1, h))
        cache = {'f':np.zeros((n,T,h)),'i':np.zeros((n,T,h)),'cbar':np.zeros((n,T,h)),
                  'o':np.zeros((n,T,h)),'z':np.zeros((n,T,h+f))}
        for t in range(T):
            zt = np.concatenate([Hs[:,t,:], X[:,t,:]], axis=1)
            ft = self.sigmoid(zt@self.Wf + self.bf)
            it = self.sigmoid(zt@self.Wi + self.bi)
            cbar = np.tanh(zt@self.Wc + self.bc)
            ot = self.sigmoid(zt@self.Wo + self.bo)
            Cs[:,t+1,:] = ft*Cs[:,t,:] + it*cbar
            Hs[:,t+1,:] = ot*np.tanh(Cs[:,t+1,:])
            cache['f'][:,t,:]=ft; cache['i'][:,t,:]=it; cache['cbar'][:,t,:]=cbar
            cache['o'][:,t,:]=ot; cache['z'][:,t,:]=zt
        h_last = Hs[:,T,:]
        logits = h_last@self.Wy + self.by
        if self.task=='classification':
            pred = self.sigmoid(logits)
        else:
            pred = logits
        return pred.ravel(), (Hs, Cs, cache, h_last)

    def backward(self, X, y, pred, state):
        n, T, f = X.shape
        h = self.h
        Hs, Cs, cache, h_last = state
        y = y.reshape(-1,1); pred_col = pred.reshape(-1,1)
        if self.task=='classification':
            dlogits = (pred_col - y)/n           # BCE + sigmoid combined grad
        else:
            dlogits = 2*(pred_col - y)/n          # MSE grad

        grads = {p: np.zeros_like(getattr(self,p)) for p in self.params}
        grads['Wy'] = h_last.T @ dlogits
        grads['by'] = dlogits.sum(axis=0)
        dh_next = dlogits @ self.Wy.T
        dc_next = np.zeros((n,h))

        for t in reversed(range(T)):
            ft = cache['f'][:,t,:]; it = cache['i'][:,t,:]; cbar = cache['cbar'][:,t,:]
            ot = cache['o'][:,t,:]; zt = cache['z'][:,t,:]
            c_t = Cs[:,t+1,:]; c_prev = Cs[:,t,:]
            tanh_c = np.tanh(c_t)

            dh = dh_next
            do = dh * tanh_c
            dc = dc_next + dh * ot * (1 - tanh_c**2)
            df = dc * c_prev
            di = dc * cbar
            dcbar = dc * it
            dc_prev = dc * ft

            do_raw = do * ot*(1-ot)
            df_raw = df * ft*(1-ft)
            di_raw = di * it*(1-it)
            dcbar_raw = dcbar * (1-cbar**2)

            grads['Wf'] += zt.T @ df_raw; grads['bf'] += df_raw.sum(axis=0)
            grads['Wi'] += zt.T @ di_raw; grads['bi'] += di_raw.sum(axis=0)
            grads['Wc'] += zt.T @ dcbar_raw; grads['bc'] += dcbar_raw.sum(axis=0)
            grads['Wo'] += zt.T @ do_raw; grads['bo'] += do_raw.sum(axis=0)

            dz = df_raw@self.Wf.T + di_raw@self.Wi.T + dcbar_raw@self.Wc.T + do_raw@self.Wo.T
            dh_next = dz[:, :h]
            dc_next = dc_prev
        return grads

    def adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for p in self.params:
            g = grads[p]
            self.m[p] = beta1*self.m[p] + (1-beta1)*g
            self.v[p] = beta2*self.v[p] + (1-beta2)*(g**2)
            mhat = self.m[p]/(1-beta1**self.t)
            vhat = self.v[p]/(1-beta2**self.t)
            setattr(self, p, getattr(self,p) - self.lr*mhat/(np.sqrt(vhat)+eps))

    def fit(self, X, y, epochs=30, batch_size=64, verbose=False):
        n = X.shape[0]
        rng = np.random.default_rng(0)
        for ep in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                b = idx[start:start+batch_size]
                pred, state = self.forward(X[b])
                grads = self.backward(X[b], y[b], pred, state)
                self.adam_step(grads)
            if verbose and ep%10==0:
                pred_all,_ = self.forward(X)
                if self.task=='classification':
                    eps=1e-9
                    loss = -np.mean(y*np.log(pred_all+eps)+(1-y)*np.log(1-pred_all+eps))
                else:
                    loss = np.mean((pred_all-y)**2)
                print(f"  epoch {ep} loss {loss:.5f}")

    def predict(self, X):
        pred,_ = self.forward(X)
        return pred

# ---- quick smoke test ----
if __name__ == '__main__':
    rng = np.random.default_rng(1)
    n, T, f = 300, 30, 5
    X = rng.normal(size=(n,T,f))
    y_reg = X[:,-1,0]*0.5 + X[:,:,1].mean(axis=1) + rng.normal(scale=0.05,size=n)
    y_clf = (X[:,-1,0] + X[:,:,2].sum(axis=1) > 0).astype(float)

    m1 = NumpyLSTM(f, hidden=8, task='regression', lr=0.02)
    m1.fit(X, y_reg, epochs=40, verbose=True)
    pred = m1.predict(X)
    print("train MSE:", np.mean((pred-y_reg)**2), "vs var(y):", np.var(y_reg))

    m2 = NumpyLSTM(f, hidden=8, task='classification', lr=0.02)
    m2.fit(X, y_clf, epochs=40, verbose=True)
    pred2 = m2.predict(X)
    acc = ((pred2>0.5).astype(float)==y_clf).mean()
    print("train acc:", acc)
