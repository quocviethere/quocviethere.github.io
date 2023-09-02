# AIO2023 NOTE | 230830 | KNN Warm-up

Instructor: Dr. Vinh

Deliverables: 
- Slide: <https://drive.google.com/file/d/1kpFcQZCx39MemG4KGU_P1q2PLGUz0omR/view?usp=drive_link>
- Code: <https://drive.google.com/drive/folders/1DClBdQuOwDWf1zpImKjAaXFur8uxPm7F> 

Further Reading: 
1. <https://machinelearningcoban.com/2017/01/08/knn/>
2. <https://www.youtube.com/watch?v=HVXime0nQeI&ab_channel=StatQuestwithJoshStarmer> 

Last edited time: September 2, 2023 11:04 AM

# Introduction to KNN

KNN is an unsupervised learning approach.

![KNN steps](KNN%209ab0cb217f23436785a130f778bb220d/Screen_Shot_2023-08-29_at_20.20.43.png)

![KNN_Working.gif](KNN%209ab0cb217f23436785a130f778bb220d/KNN_Working.gif)

## Distance

Distance càng nhỏ → thuộc tính càng giống nhau

Euclidean distance

$$
d(x,y) = \sqrt{\sum_{i=1}^{n}(x_{i}-y_{i})^{2}}
$$

Mahhatan distance

Minkowsky distance

Chebyshev distance

## Procedure

**Procedure**

-   Data processing and select $K$
-   Compute distances
-   Sort distances
-   Get top $K$ points
-   Vote and return majority

There are multiple ways to choose $K$, but conventionally $K = \sqrt{N}$ where $N$ is the number of samples (data points).

## Case Study

**Case study: Iris dataset**

$$
d = \sqrt{(x_{1}^{(train)}-x_{1}^{(test)})^{2} + (x_{2}^{(train)}-x_{2}^{(test)})^{2}}
$$

| Petal Length  | Label | Distance | Distance |
|---------------|-------|----------|----------|
| 1.4           | 0     | 1        | 1        |
| 1             | 0     | 1.4      | 1.4      |
| 1.5           | 0     | 0.9      | 0.9      |
| 3.1           | 1     | 0.7      | 0.7      |
| 3.7           | 1     | 1.3      | 1.3      |
| 4.1           | 1     | 1.7      | 1.7      |
| New data: 2.4 | 1     | $k=1$    | $k=3$    |

For $k=1$, the smallest distance is 0.7, therefore the label of the new data is the same as the label with the smallest distance, which is 1.

For $k=3$, the 3 smallest distances are 0.7, 0.9, and 1, two of which data points have label of 0, by majority voting, the label of the new data is now 0.

``` python
distances = np.sqrt(np.sum((x_data-x_test)**2,axis = 1))
```

# Data Normalization

![](KNN%209ab0cb217f23436785a130f778bb220d/Screen_Shot_2023-08-29_at_21.08.13.png)

![](KNN%209ab0cb217f23436785a130f778bb220d/Screen_Shot_2023-08-29_at_21.07.37.png)

-   The range of the values is different
-   Different unit measurement

$$
x = \dfrac{x-\mu}{\sigma}
$$

# Application

## Text classification

![Screen Shot 2023-08-29 at 21.22.50.png](KNN%209ab0cb217f23436785a130f778bb220d/Screen_Shot_2023-08-29_at_21.22.50.png)

Vocabulary size $|V| = 9$

``` python
vectorizer = CountVectorizer()
corpus = ["góp gió gặt bão",
          "có làm mới có ăn",
          "đất lành chim đậu",
          "ăn cháo đá bát",
          "gậy ông đập lưng ông",
          "qua cầu rút ván"]
X = vectorizer.fit_transform(corpus)
```

# Entropy

Suppose we have 10 balls, 9 of which are red and there is only 1 blue ball. Let $A$ be the event that the ball is red and $B$ be the event that the ball is blue, we have that:

$$P(E_1) = 0.9 \text{ and } P(E_2) = 0.1$$

Since the probability of $A$ happening is larger, we would become more surprised if $B$ happens. In other words,

$$surprise(E) = \dfrac{1}{P(E)}$$

when $P(E) \rightarrow 0$, $surprise(E)$ would go to infinity, on the other hand, when $P(E) \rightarrow 1$, $surprise(E) = 1$, meaning we are not surprised at all. Intuitively, when we are not surprised at all we want the value to be equal to 0, therefore we will take the $log$ of $surprise(E)$ since $log(1) = 0$, the formula can now be written as:

$$
surprise(E) = log \left( \dfrac{1}{P(E)}\right)
$$

**Expectation:**

$$
\mathbb{E}(X) = \sum xp(x)
$$

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

Entropy:

$$H(X) = - \underset{x \in \mathcal{X}}{\sum}p(x)logp(x)$$
