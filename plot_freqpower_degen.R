require(ggplot2)
require(data.table)

# Unconditional data
d_uncond_gold <- fread("plot/degen/unconditional_gold.model=gpt2.density.csv")
d_uncond_gold$type <- "gold"

d_uncond_ps <- fread("plot/degen/unconditional_puresampling_large.model=gpt2.density.csv")
d_uncond_ps$type <- "pure-sampling"

d_uncond <- rbindlist(list(d_uncond_gold, d_uncond_ps))

# plot
p <- ggplot(d_uncond, aes(freq, power)) +
  geom_smooth(aes(linetype = type, fill = type, colour = type))