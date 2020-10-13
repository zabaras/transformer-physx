---
title: Transformer-PhysX
layout: page
feature_text: |
  ## Transformer-PhysX
  #### Transformers for modeling physical systems
---
Transformers are widely used in neural language processing due to their ability to model longer-term dependencies in text.
Although these models achieve state-of-the-art performance for many language related tasks, their applicability outside of the neural language processing field has been minimal.
This library is designed to deploy transformer models for the prediction of dynamical systems representative of physical phenomena.

{% include carousel.html height="40" unit="%" duration="8" %}

### Publications

- Transformers for Modeling Physical Systems [[ArXiVs](https://arxiv.org/abs/2010.03957)]


### Highlights

Prediction of ordinary differential equations from any intial state.
<video width="1200" height="300" controls>
  <source src="assets/imgs/lorenz0.mp4" type="video/mp4">
  <source src="assets/imgs/lorenz0.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

---

Transformer with Koopman based embeddings (Transformer-KM) is able to out perform embedding methods and ConvLSTM for prediction of flow around a cylinder at different Reynolds numbers.
<figure>
<video width="1200" height="200" controls>
  <source src="assets/imgs/cylinder0.mp4" type="video/mp4">
  <source src="assets/imgs/cylinder0.webm" type="video/webm">
  Your browser does not support the video tag.
</video>
<figcaption>Reynolds number 133</figcaption>
</figure>
<figure>
<video width="1200" height="200" controls>
  <source src="assets/imgs/cylinder1.mp4" type="video/mp4">
  <source src="assets/imgs/cylinder1.webm" type="video/webm">
  Your browser does not support the video tag.
</video>
<figcaption>Reynolds number 433</figcaption>
</figure>

---

Prediction of 3D reaction diffusion system with random initial condition.
<video width="1200" height="400" controls>
  <source src="assets/imgs/grayscott0.mp4" type="video/mp4">
  <source src="assets/imgs/grayscott0.webm" type="video/webm">
  Your browser does not support the video tag.
</video>
<video width="1200" height="400" controls>
  <source src="assets/imgs/grayscott1.mp4" type="video/mp4">
  <source src="assets/imgs/grayscott1.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

### Cite

Found this helpful or interesting? Cite us with:

```markdown
    @article{geneva2020transformers,
      title={Transformers for Modeling Physical Systems},
      author={Geneva, Nicholas and Zabaras, Nicholas},
      journal={arXiv preprint arXiv:2010.03957},
      year={2020}
    }
```

### Support or Contact
This is currently under the core development of [Nicholas Geneva](https://nicholasgeneva.com/) from Notre Dame, USA.
Having some issues or questions regarding the code? Create an issue on the repository! 
