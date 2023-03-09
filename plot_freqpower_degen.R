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


# Unconditional, sampling with temperature 0.1 to 0.9
d_uncond_sampling <- data.table(freq=numeric(),
                                power=numeric(),
                                temperature=character())
for (t in 1:9) {
  tmp <- fread(paste0("plot/degen/unconditional_sampling_t=0.", t, "_large.model=gpt2.density.csv"))
  tmp$temperature <- paste0("0.", t)
  d_uncond_sampling <- rbindlist(list(d_uncond_sampling, tmp))
}
# combine with gold
d_uncond_gold <- fread("plot/degen/unconditional_gold.model=gpt2.density.csv")
d_uncond_gold$temperature <- "gold"
d_uncond_sampling <- rbindlist(list(d_uncond_sampling, d_uncond_gold))

p <- ggplot(d_uncond_sampling, aes(freq, power)) +
  geom_smooth(aes(linetype = temperature, fill = temperature, colour = temperature))
ggsave("plot/degen/uncond_smpl_temp1to9.pdf", plot=p)


# Unconditional, top k with diff values
d_uncond_topk <- data.table(freq=numeric(),
                            power=numeric(),
                            k=character())
for (k in c(5, 10, 20, 40, 80, 160, 320, 640)) {
  tmp <- fread(paste0("plot/degen/unconditional_topk_k=", k, "_large.model=gpt2.density.csv"))
  if (k>=100) {
    tmp$k <- as.character(k)
  } else if (k>=10) {
    tmp$k <- paste0("0", k)
  } else {
    tmp$k <- paste0("00", k)
  }
  d_uncond_topk <- rbindlist(list(d_uncond_topk, tmp))
}
# combine with gold
d_uncond_gold <- fread("plot/degen/unconditional_gold.model=gpt2.density.csv")
d_uncond_gold$k <- "gold"
d_uncond_topk <- rbindlist(list(d_uncond_topk, d_uncond_gold))

p <- ggplot(d_uncond_topk, aes(freq, power)) +
  geom_smooth(aes(linetype = k, fill = k, colour = k))
ggsave("plot/degen/uncond_topk_k5to640.pdf", plot=p)


# Unconditional, top p, p from 0.1 to 0.9
d_uncond_topp <- data.table(freq=numeric(),
                            power=numeric(),
                            p=character())
for (p in 1:9) {
  tmp <- fread(paste0("plot/degen/unconditional_topp_p=0.", p, "_large.model=gpt2.density.csv"))
  tmp$p <- paste0("0.", p)
  d_uncond_topp <- rbindlist(list(d_uncond_topp, tmp))
}
# combine with gold
d_uncond_gold <- fread("plot/degen/unconditional_gold.model=gpt2.density.csv")
d_uncond_gold$p <- "gold"
d_uncond_topp <- rbindlist(list(d_uncond_topp, d_uncond_gold))

p <- ggplot(d_uncond_topp, aes(freq, power)) +
  geom_smooth(aes(linetype = p, fill = p, colour = p))
ggsave("plot/degen/uncond_topp_p1to9.pdf", plot=p)