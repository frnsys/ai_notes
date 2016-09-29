good explanation of conditional in/dependence here: <https://medium.com/@akelleh/understanding-bias-a-pre-requisite-for-trustworthy-results-ee590b75b1be#.1br10vwge>


Given the simple graph:

```
sprinker ->
           -> wet
rain     ->
```

that is the sidewalk being wet is a _common effect_ of it raining and the sprinkler being on.

If we see that the sidewalk is wet and know that it doesn't rain, that tells us something about whether or not the sprinkler was on - that is, when the sidewalk is wet, knowing something about whether or not it rained affects makes the sprinkler being on more likely.

Even though there's no direct connection between the sprinkler and rain (if we just had a dataset of the two, we'd see no correlation), there is a relationship _conditioned_ on the sidewalk being wet (that is, if we just looked at days where the sidewalk was wet, we _would_ see a (negative) correlation between the two). So while rain and the sprinkler are _independent_, they are not _conditionally independent_ (independence doesn't imply and is not implied by conditional independence).

That is: "_conditioning on a common effect results in two causes becoming correlated_, even if they were uncorrelated originally" (this is called "Berkson's paradox")


Now consider:

```
           -> traffic
disaster ->
           -> alarm
```

here a disaster is a _common cause_ of traffic and the alarm going off. Even though there is no causal relationship between the traffic and the alarm, we would see correlations between the two due to this common cause (this bias is called "confounding". We can remove this spurious correlation by conditioning, i.e. looking at data without disasters, and then we'd see that traffic and the alarm are uncorrelated.

If I know there was a disaster, knowing the alarm went off doesn't give any more information about traffic (that information is already provided by knowing there was a disaster).

To sum up:

- bias comes from conditioning when you condition on a _common effect_; it goes away when you don't condition
- bias comes from _not_ conditioning on a _common cause_; it goes away when you _do_ condition

The "back-door criterion" says given "any sufficiently complete" picture of the world (which is not easy to achieve), we shouldn't condition on a common effect, and we shouldn't condition on common causes.