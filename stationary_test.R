require("data.table")
require("ggplot2")
require("stringr")
require("progress") # tqdm for R

# require("fpp")
# require("forecast")


# Read entropy data
file_name <- "data/small-117M.test.model=gpt2.nll"
file_conn <- file(file_name, "r")
lines <- readLines(file_conn)

# pb <- progress_bar$new(format = "  downloading [:bar] :percent eta: :eta",
#                        total = 100)
# pb$tick(0)
# data <- data.table()
# for (i in 1:length(lines)) {
#   entropy = as.numeric(str_split(lines[i], " ")[[1]])
#   seriesID = rep(i, length(entropy))
#   data <- rbindlist(list(data, data.table(seriesID = seriesID, entropy = entropy)))
#   pb$tick()
# }
## 不好用，3/2/2023
