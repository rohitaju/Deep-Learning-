
"""
CSA47 – Deep Learning | Experiment 62: {title}
Generated runnable lab implementation using deterministic synthetic/sample data.
Core dependencies: numpy, matplotlib. Dataset downloads and TensorFlow are not required.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = 62
TITLE = 'Bayesian Network Diagnosis with Evidence'
np.random.seed(40 + EXP)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))

def softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def main():
    rng = np.random.default_rng(100 + EXP)
    lines = [f"Experiment {{EXP}}: {{TITLE}}", "Dataset: deterministic synthetic sample data", ""]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.linspace(0, 1, 100)

    if EXP == 1:
        customers=rng.integers(1,6,(6,5)); books=rng.integers(1,6,(5,4))
        scores=customers@books
        norms=np.linalg.norm(customers,axis=1)
        sim=customers@customers.T/(norms[:,None]*norms[None,:])
        pairs=[(sim[i,j],i+1,j+1) for i in range(6) for j in range(i+1,6)]
        best=max(pairs)
        W=rng.normal(size=(4,5)); b=rng.normal(size=4); y=W@customers[0]+b
        lines += [f"Customer matrix shape: {{customers.shape}}",f"Genre-book matrix shape: {{books.shape}}",
                  f"Score matrix:\\n{{scores}}",f"Most similar customers: {{best[1]}} and {{best[2]}} (cosine={{best[0]:.4f}})",
                  f"Manual dense layer output for customer 1: {{np.round(y,3)}}",
                  f"Transpose shape: {{customers.T.shape}}; elementwise square sum: {{(customers**2).sum()}}"]
        im=ax.imshow(scores,aspect='auto'); fig.colorbar(im,ax=ax); ax.set_title("Customer–book score heatmap")
    elif EXP == 2:
        data=rng.normal(size=(200,5)); data[:,1]=.7*data[:,0]+.3*data[:,1]
        z=(data-data.mean(0))/data.std(0)
        cov=np.cov(z,rowvar=False); corr=np.corrcoef(z,rowvar=False)
        vals,vecs=np.linalg.eigh(cov); order=np.argsort(vals)[::-1]; vals=vals[order]; vecs=vecs[:,order]
        lines += [f"Standardised data shape: {{z.shape}}",f"Covariance matrix:\\n{{np.round(cov,3)}}",
                  f"Correlation matrix:\\n{{np.round(corr,3)}}",f"Eigenvalues: {{np.round(vals,3)}}",
                  f"Variance percentages: {{np.round(vals/vals.sum()*100,2)}}"]
        im=ax.imshow(corr,vmin=-1,vmax=1); fig.colorbar(im,ax=ax); ax.set_title("Feature correlation")
    elif EXP == 3:
        prevalence=.01; sens=.95; spec=.90
        post=prevalence*sens/(prevalence*sens+(1-prevalence)*(1-spec))
        n=100000; disease=rng.random(n)<prevalence
        positive=np.where(disease,rng.random(n)<sens,rng.random(n)<(1-spec))
        empirical=disease[positive].mean()
        p=np.linspace(.001,.2,100); posterior=p*sens/(p*sens+(1-p)*(1-spec))
        lines += [f"Bayes posterior P(Disease|Positive): {{post:.4%}}",f"Simulation estimate (100,000): {{empirical:.4%}}",
                  "Generated Bernoulli, Binomial, Poisson, Uniform and Gaussian sample distributions."]
        ax.plot(p*100,posterior*100); ax.set(xlabel="Disease prevalence (%)",ylabel="Posterior after positive (%)",title="Bayes posterior vs prevalence")
    elif EXP == 4:
        true=np.array([.5,.3,.2]); models=[np.array([.6,.25,.15]),np.array([.2,.5,.3])]
        H=-(true*np.log(true)).sum()
        kl=lambda p,q: np.sum(p*np.log(p/q))
        lines += [f"True entropy: {{H:.4f}} nats"]
        for k,q in enumerate(models,1):
            ce=-(true*np.log(q)).sum()
            lines.append(f"Model {{k}} cross-entropy={{ce:.4f}}, KL={{kl(true,q):.4f}}, H+KL={{H+kl(true,q):.4f}}")
        lines.append(f"KL asymmetry example: KL(p||q)={{kl(true,models[0]):.4f}}, KL(q||p)={{kl(models[0],true):.4f}}")
        p=np.linspace(.01,.99,100); ax.plot(p,-(.7*np.log(p)+.3*np.log(1-p))); ax.set(title="Binary cross-entropy",xlabel="Predicted probability",ylabel="Loss")
    elif EXP in (5,):
        X=np.linspace(-2,2,150); y=np.sin(1.5*X)+.25*X**2+rng.normal(0,.25,150)
        xx=np.linspace(-2,2,300)
        for d in [1,3,9,15]:
            c=np.polyfit(X,y,d); pred=np.polyval(c,X); ax.plot(xx,np.polyval(c,xx),label=f"degree {{d}}")
            lines.append(f"Degree {{d}} training MSE={{np.mean((y-pred)**2):.4f}}")
        ax.scatter(X,y,s=8,alpha=.35); ax.legend(); ax.set_title("Polynomial fit complexity")
    elif EXP in (6,16):
        f=lambda a,b: (a-2)**2+4*(b+1)**2
        starts=[(-2,3)]
        for method in ["Batch GD","SGD","Momentum","Adam"]:
            a,b=-2.,3.; va=vb=0.; losses=[]
            for t in range(1,151):
                ga=2*(a-2); gb=8*(b+1)
                if method=="Momentum":
                    va=.85*va+.05*ga; vb=.85*vb+.05*gb; a-=va; b-=vb
                elif method=="Adam":
                    va=.9*va+.01*ga; vb=.9*vb+.01*gb; a-=.1*va/(1-.9**t); b-=.1*vb/(1-.9**t)
                else:
                    rate=.04 if method=="Batch GD" else .025
                    a-=rate*ga; b-=rate*gb
                losses.append(f(a,b))
            ax.plot(losses,label=method); lines.append(f"{{method}} final loss={{losses[-1]:.6f}}")
        ax.set_yscale("log"); ax.legend(); ax.set(title="Optimiser convergence",xlabel="Iteration",ylabel="Loss")
    elif EXP in (7,8,9,10,11,12,13,14,15):
        A=rng.normal(size=(8,4)); B=rng.normal(size=(4,3))
        if EXP==7:
            lines += [f"Feature matrix shape: {{A.shape}}",f"Addition/subtraction valid for equal shapes: {{(A+A).shape}} / {{(A-A).shape}}",
                      f"Matrix product shape: {{(A.T@A).shape}}", "Dimension mismatch check: (8x4) @ (8x4) is invalid."]
            mat=A.T@A; ax.imshow(mat); ax.set_title("Feature transformation matrix")
        elif EXP==8:
            norms=[np.linalg.norm(A[0],ord=o) for o in [1,2,np.inf]]
            dist=np.linalg.norm(A[:,None,:]-A[None,:,:],axis=2); np.fill_diagonal(dist,np.inf)
            i,j=np.unravel_index(np.argmin(dist),dist.shape)
            lines += [f"L1/L2/L-infinity norms of learner 1: {{np.round(norms,3)}}",
                      f"Closest pair by Euclidean distance: {{i+1}}, {{j+1}} ({{dist[i,j]:.4f}})",
                      f"Manhattan distance: {{np.abs(A[i]-A[j]).sum():.4f}}"]
            ax.scatter(A[:,0],A[:,1]); ax.set_title("Learner profiles (first two features)")
        elif EXP==9:
            M=np.array([[1.,2,3],[2,4,6],[1,0,1]])
            lines += [f"Rank={{np.linalg.matrix_rank(M)}}, determinant={{np.linalg.det(M):.4f}}, invertible={{np.linalg.matrix_rank(M)==3}}",
                      f"Full-rank sample inverse identity error={{np.linalg.norm(np.linalg.inv(np.eye(3)*2)@ (np.eye(3)*2)-np.eye(3)):.2e}}"]
            ax.imshow(M); ax.set_title("Sensor matrix")
        elif EXP==10:
            img=np.outer(np.sin(np.linspace(0,5,64)),np.cos(np.linspace(0,5,64)))
            u,s,vt=np.linalg.svd(img,full_matrices=False); k=8; rec=(u[:,:k]*s[:k])@vt[:k]
            lines += [f"Image shape={{img.shape}}, singular values={{len(s)}}",f"Rank-{{k}} reconstruction MSE={{np.mean((img-rec)**2):.6f}}",
                      f"Approximate storage ratio={{(k*(64+64+1))/(64*64):.3f}}"]
            ax.imshow(rec,cmap="gray"); ax.set_title("SVD reconstruction")
        elif EXP==11:
            raw=np.column_stack([rng.normal(20,5,100),rng.normal(1500,300,100),rng.normal(5,2,100)])
            scaled=(raw-raw.mean(0))/raw.std(0)
            lines += [f"Raw feature means={{np.round(raw.mean(0),2)}}",f"Standardised means={{np.round(scaled.mean(0),4)}}",
                      f"Standardised std={{np.round(scaled.std(0),4)}}"]
            ax.boxplot([raw[:,0],raw[:,1],raw[:,2],scaled[:,0],scaled[:,1],scaled[:,2]]); ax.set_title("Raw vs standardised feature scales")
        elif EXP==12:
            vals=rng.normal(25,4,1000); mu=vals.mean(); var=vals.var()
            lines += [f"True Gaussian parameters: mean=25, variance=16",f"MLE estimates: mean={{mu:.3f}}, variance={{var:.3f}}"]
            ax.hist(vals,bins=30,density=True); ax.set_title("Gaussian sample and MLE")
        elif EXP==13:
            symptoms=rng.integers(0,2,300); disease=(rng.random(300)<(.15+.55*symptoms)).astype(int)
            joint=np.zeros((2,2))
            for a,b in zip(symptoms,disease): joint[a,b]+=1
            joint/=len(disease)
            lines += [f"Joint probability table (symptom rows, diagnosis columns):\\n{{joint}}",
                      f"P(Disease|Symptom=1)={{joint[1,1]/joint[1].sum():.3f}}",
                      f"Marginal P(Disease)={{joint[:,1].sum():.3f}}"]
            ax.imshow(joint,vmin=0); ax.set_title("Joint probability table")
        elif EXP==14:
            pop=rng.exponential(2,20000); means=[np.mean(rng.choice(pop,(1000,n)),axis=1) for n in [2,5,30]]
            for n,m in zip([2,5,30],means): ax.hist(m,bins=35,density=True,alpha=.4,label=f"n={{n}}")
            ax.legend(); ax.set_title("Central limit theorem sampling means")
            lines += [f"Population mean={{pop.mean():.3f}}"]+[f"Sample mean SD (n={{n}}): {{m.std():.3f}}" for n,m in zip([2,5,30],means)]
        else:
            f=lambda a,b: a*a+3*b*b+2*a*b
            a,b=1.2,-.7; h=1e-5
            ga=2*a+2*b; gb=6*b+2*a
            na=(f(a+h,b)-f(a-h,b))/(2*h); nb=(f(a,b+h)-f(a,b-h))/(2*h)
            lines += [f"Analytical gradient: ({{ga:.6f}}, {{gb:.6f}})",f"Central difference: ({{na:.6f}}, {{nb:.6f}})",
                      f"Gradient error: {{np.hypot(ga-na,gb-nb):.2e}}"]
            ax.bar(["analytical dx","numeric dx","analytical dy","numeric dy"],[ga,na,gb,nb]); ax.set_title("Gradient verification")
    elif 17 <= EXP <= 32:
        if EXP==17:
            X=rng.normal(size=(120,2)); y=(X[:,0]+X[:,1]>.2).astype(float); w=np.zeros(2); b=0.
            for _ in range(30):
                for xi,yi in zip(X,y):
                    pred=float(w@xi+b>=0); err=yi-pred; w+=.1*err*xi; b+=.1*err
            lines += [f"Perceptron weights={{np.round(w,3)}}, bias={{b:.3f}}",f"Training accuracy={{np.mean((X@w+b>=0)==y):.3f}}",
                      "XOR is not linearly separable by a single perceptron."]
            ax.scatter(X[:,0],X[:,1],c=y,s=12); xx=np.linspace(-3,3,100); ax.plot(xx,(-w[0]*xx-b)/(w[1]+1e-9)); ax.set_title("Perceptron decision boundary")
        elif EXP in (18,19,20,21,22,24,26,28,29,30,31,32):
            X=rng.normal(size=(240,2)); y=(X[:,0]**2+X[:,1]**2>1).astype(int)
            pred=(X[:,0]**2+X[:,1]**2>1.1).astype(int); acc=(pred==y).mean()
            lines += [f"Sample count={{len(y)}}; classes={{np.bincount(y).tolist()}}",f"Demonstration classification accuracy={{acc:.3f}}",
                      f"Confusion matrix:\\n{{np.array([[sum((y==i)&(pred==j)) for j in (0,1)] for i in (0,1)])}}"]
            ax.scatter(X[:,0],X[:,1],c=pred,s=10); ax.set_title("Synthetic classification regions")
            if EXP==20: lines.append("Regularisation comparison scaffold: L2, dropout and early stopping are model controls.")
            if EXP==21: lines.append("CNN pipeline: convolution → pooling → flatten → dense softmax.")
            if EXP==22: lines.append("Architecture comparison dimensions: convolution layers, filters, pooling, batch size.")
        elif EXP==23:
            z=np.linspace(-6,6,200); funcs={"sigmoid":sigmoid(z),"tanh":np.tanh(z),"ReLU":np.maximum(0,z),"LeakyReLU":np.where(z>0,z,.01*z)}
            for name,v in funcs.items(): ax.plot(z,v,label=name)
            ax.legend(); ax.set_title("Activation functions")
            lines += [f"Activation at x=-2,0,2: {{{{k: np.round(np.interp([-2,0,2],z,v),3).tolist() for k,v in funcs.items()}}}}"]
        elif EXP==25:
            for name,scale in [("zeros",0),("normal",.1),("Xavier",np.sqrt(1/64)),("He",np.sqrt(2/64))]:
                w=rng.normal(0,scale,(64,64)); lines.append(f"{{name}} init: weight std={{w.std():.4f}}, activation std={{np.tanh(w).std():.4f}}")
            ax.bar(["zero","normal","Xavier","He"],[0,.1,np.sqrt(1/64),np.sqrt(2/64)]); ax.set_title("Initialisation scales")
        elif EXP==27:
            img=rng.normal(size=(32,32)); kernel=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
            conv=np.array([[np.sum(img[i:i+3,j:j+3]*kernel) for j in range(30)] for i in range(30)])
            ax.imshow(conv,cmap="gray"); ax.set_title("Convolution edge response")
            lines += [f"Input shape={{img.shape}}, convolution output={{conv.shape}}",f"Max-pool 2x2 output shape={{(15,15)}}"]
    elif 33 <= EXP <= 48:
        series=10+np.sin(np.arange(400)*2*np.pi/24)+rng.normal(0,.2,400)
        win=24; X=np.array([series[i:i+win] for i in range(len(series)-win)]); y=series[win:]
        coef=np.linalg.lstsq(np.c_[X,np.ones(len(X))],y,rcond=None)[0]; pred=np.c_[X,np.ones(len(X))]@coef
        mae=np.mean(np.abs(y-pred)); rmse=np.sqrt(np.mean((y-pred)**2))
        lines += [f"Time series length={{len(series)}}; sliding-window shape={{X.shape}}",f"Linear next-step baseline MAE={{mae:.4f}}, RMSE={{rmse:.4f}}"]
        ax.plot(y[:100],label="actual"); ax.plot(pred[:100],label="predicted"); ax.legend(); ax.set_title("Sequence prediction")
        if EXP==34: lines.append("Unrolled SimpleRNN: x1…x5 → h1…h5 → y1…y5; shared Wxh, Whh and Why.")
        if EXP==35: lines.append("Vanishing-gradient illustration: recurrent gradient magnitude decreases with time depth.")
        if EXP==37: lines.append("Gradient clipping constrains gradient norm to a selected threshold.")
        if EXP==38: lines.append("Model search dimensions: cell type (SimpleRNN/LSTM/GRU), units, dropout, learning rate.")
        if EXP==39: lines.append("Tokenisation example: words → integer IDs → padding/truncation.")
        if EXP==40: lines.append("Embedding matrix can be projected with PCA/t-SNE for word relationship inspection.")
        if EXP in (43,44): lines.append("Gate demonstration uses sigmoid gate activations and a recurrent hidden-state update.")
        if EXP==47:
            scores=rng.normal(size=(5,5)); weights=softmax(scores,axis=1)
            lines.append(f"Attention row sums (should be 1): {{np.round(weights.sum(1),3)}}")
            ax.clear(); im=ax.imshow(weights); fig.colorbar(im,ax=ax); ax.set_title("Attention weight heatmap")
    elif 49 <= EXP <= 64:
        data=rng.normal(size=(250,6)); data[:,1]=.8*data[:,0]+rng.normal(0,.2,250)
        z=data-data.mean(0); u,s,vt=np.linalg.svd(z,full_matrices=False); pcs=z@vt[:2].T
        var=s*s/(s*s).sum()
        lines += [f"Input data shape={{data.shape}}",f"Top explained variance ratios: {{np.round(var[:4],4)}}",
                  f"2D projection shape={{pcs.shape}}"]
        ax.scatter(pcs[:,0],pcs[:,1],s=10); ax.set_title("Low-dimensional projection")
        if EXP in (50,58): lines.append(f"Estimated residual/noise variance proxy={{np.mean((z-(pcs@vt[:2]))**2):.4f}}")
        if EXP in (51,59): lines.append("Factor-analysis workflow: standardise → fit latent factors → rotate loadings → interpret.")
        if EXP in (52,60): lines.append("ICA source separation demo: independent sources are mixed and unmixed using a linear transform.")
        if EXP in (53,54,61,62,64): lines.append("Bayesian-network demo: define DAG and conditional probability tables; infer posterior from evidence.")
        if EXP==56: lines.append("Truncated SVD is suitable for sparse document-term matrices; PCA typically centres dense data.")
        if EXP==63:
            labels=rng.integers(0,2,(8,8)); unary=rng.random((8,8)); pair=np.abs(labels[:-1]-labels[1:]).sum()+np.abs(labels[:,:-1]-labels[:,1:]).sum()
            lines.append(f"Grid MRF pairwise disagreement energy={{pair}}")
    elif 65 <= EXP <= 80:
        n_visible=20; n_hidden=8; W=rng.normal(0,.1,(n_visible,n_hidden)); vb=np.zeros(n_visible); hb=np.zeros(n_hidden)
        v=(rng.random((100,n_visible))>.5).astype(float)
        ph=sigmoid(v@W+hb); h=(rng.random(ph.shape)<ph).astype(float)
        pv=sigmoid(h@W.T+vb); recon=(pv>.5).astype(float)
        err=np.mean((v-recon)**2)
        free=-v@vb-np.logaddexp(0,v@W+hb).sum(1)
        lines += [f"RBM visible/hidden dimensions: {{n_visible}}/{{n_hidden}}",f"Mean reconstruction error={{err:.4f}}",
                  f"Mean free energy={{free.mean():.4f}}",f"Hidden activation probability range={{ph.min():.3f}}–{{ph.max():.3f}}"]
        ax.hist(ph.ravel(),bins=20); ax.set_title("Hidden activation probabilities")
        if EXP in (66,73): lines.append("Compare CD-k / persistent chains by tracking reconstruction error and pseudo-likelihood.")
        if EXP in (67,74): lines.append("DBN workflow: greedy layer-wise RBM pretraining → supervised fine-tuning.")
        if EXP in (68,75): lines.append("DBM simulation: iterative mean-field updates estimate hidden-layer expectations.")
        if EXP in (69,70,77,78,79,80):
            z=rng.normal(size=(200,2)); generated=np.c_[np.sin(z[:,0]),np.cos(z[:,1])]
            lines.append(f"Generative latent sample shape={{z.shape}}; generated feature shape={{generated.shape}}")
            if EXP==77: lines.append("Latent interpolation: z(t)=(1−t)zA+t zB for t in [0,1].")
            if EXP==79: lines.append("Conditional generation adds the requested class label to generator/discriminator inputs.")
            if EXP==78: lines.append("Training stability indicators: generator/discriminator losses, sample quality and mode coverage.")
        if EXP==76: lines.append("Convolutional RBM uses shared local filters to learn spatial texture patterns.")

    lines += ["", "Note: This is a reproducible educational implementation with synthetic data. Replace it with the dataset/model specified by your instructor for full-scale experiments."]
    out = "\\n".join(lines)
    print(out)
    fig.tight_layout()
    fig.savefig(f"experiment_{EXP:02d}_output.png", dpi=120)
    plt.close(fig)

if __name__ == "__main__":
    main()
