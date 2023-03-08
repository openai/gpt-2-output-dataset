require(ggplot2)
require(data.table)

# Unconditional data
d_uncond_gold <- fread("plot/degen/unconditional_gold.model=gpt2.density.csv")
d_uncond_gold$type <- "gold"

d_uncond_ps <- fread("plot/degen/unconditional_puresampling_large.model=gpt2.density.csv")
d_uncond_ps$type <- "pure-sampling"

d_uncond_k40 <- fread("plot/degen/unconditional_topk_k=40_t=0.7_large.model=gpt2.density.csv")
d_uncond_k40$type <- "topk=40"

d_uncond_k640 <- fread("plot/degen/unconditional_topk_k=640_large.model=gpt2.density.csv")
d_uncond_k640$type <- "topk=640"

d_uncond <- rbindlist(list(d_uncond_gold, d_uncond_ps, d_uncond_k40,
                           d_uncond_k640))

# Compare gold, ps, topk=40, and topk=640
p <- ggplot(d_uncond, aes(freq, power)) +
  geom_smooth(aes(linetype = type, fill = type, colour = type))
ggsave("plot/degen/uncond_gold_ps_topk40n640.pdf", plot=p)