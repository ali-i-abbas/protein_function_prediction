
library(Rmisc)
library("tidyverse")
library("ggplot2")



stats <- read.csv("results_baseline_15.csv")


stats_cc <- stats %>% filter(subontology == "cc") 

stats_cc_avg <- stats_cc %>% 
	group_by(encoding, gram_len, embedding_size) %>% 
	summarize(minF = min(f), maxF = max(f), avgF = mean(f), stdF = sd(f), GCI = (CI(f, ci = 0.95)[1] - CI(f, ci = 0.95)[2]))


stats_cc_avg[which.max(stats_cc_avg$avgF),]

stats_cc_avg$encoding <- as.character(stats_cc_avg$encoding)
stats_cc_avg$encoding[stats_cc_avg$encoding == 'oh'] <- 'O'
stats_cc_avg$encoding <- as.factor(stats_cc_avg$encoding)

stats_cc_avg$encoding <- as.character(stats_cc_avg$encoding)
stats_cc_avg$encoding[stats_cc_avg$encoding == 'ad'] <- 'T'
stats_cc_avg$encoding <- as.factor(stats_cc_avg$encoding)


stats_cc_avg <- stats_cc_avg %>% 
  mutate(algo = case_when(embedding_size == 0 ~ paste0(encoding, "-", gram_len),
                          TRUE ~ paste0(encoding, "-", gram_len, "-", embedding_size)))

# change factor order level for correct x-axis ordering
stats_cc_avg$algo <- factor(stats_cc_avg$algo, 
	levels=stats_cc_avg$algo[order(stats_cc_avg$encoding, stats_cc_avg$gram_len, stats_cc_avg$embedding_size, decreasing = TRUE)])




ggplot(stats_cc_avg, aes(x=algo, y=avgF, group=encoding, fill=encoding)) +
	geom_errorbar(aes(ymin=avgF-GCI, ymax=avgF+GCI), width = 0.5, size = 0.8, alpha = 0.7, color="red") +
	geom_bar(stat="identity", alpha = 0.7, show.legend = FALSE) +
	labs(title="", x ="", y = "Fmax", color="Encoding") +
	coord_flip(ylim=c(0.54,0.6)) +
	theme(text = element_text(size=20), axis.text.x = element_text(size=25), axis.text.y = element_text(size=15))
	
ggsave("cc.png", width = 7, height = 4)

# -------------------------------------------------------------------------------------------------------------

stats_mf <- stats %>% filter(subontology == "mf") 

stats_mf_avg <- stats_mf %>% 
	group_by(encoding, gram_len, embedding_size) %>% 
	summarize(minF = min(f), maxF = max(f), avgF = mean(f), stdF = sd(f), GCI = (CI(f, ci = 0.95)[1] - CI(f, ci = 0.95)[2]))


stats_mf_avg[which.max(stats_mf_avg$avgF),]

stats_mf_avg$encoding <- as.character(stats_mf_avg$encoding)
stats_mf_avg$encoding[stats_mf_avg$encoding == 'oh'] <- 'O'
stats_mf_avg$encoding <- as.factor(stats_mf_avg$encoding)

stats_mf_avg$encoding <- as.character(stats_mf_avg$encoding)
stats_mf_avg$encoding[stats_mf_avg$encoding == 'ad'] <- 'T'
stats_mf_avg$encoding <- as.factor(stats_mf_avg$encoding)


stats_mf_avg <- stats_mf_avg %>% 
  mutate(algo = case_when(embedding_size == 0 ~ paste0(encoding, "-", gram_len),
                          TRUE ~ paste0(encoding, "-", gram_len, "-", embedding_size)))

# change factor order level for correct x-axis ordering
stats_mf_avg$algo <- factor(stats_mf_avg$algo, 
	levels=stats_mf_avg$algo[order(stats_mf_avg$encoding, stats_mf_avg$gram_len, stats_mf_avg$embedding_size, decreasing = TRUE)])




ggplot(stats_mf_avg, aes(x=algo, y=avgF, group=encoding, fill=encoding)) +
	geom_errorbar(aes(ymin=avgF-GCI, ymax=avgF+GCI), width = 0.5, size = 0.8, alpha = 0.7, color="red") +
	geom_bar(stat="identity", alpha = 0.7, show.legend = FALSE) +
	labs(title="", x ="", y = "Fmax", color="Encoding") +
	coord_flip(ylim=c(0.34,0.42)) +
	theme(text = element_text(size=20), axis.text.x = element_text(size=25), axis.text.y = element_text(size=15))
	
ggsave("mf.png", width = 7, height = 4)

# -------------------------------------------------------------------------------------------------------------


stats_bp <- stats %>% filter(subontology == "bp") 

stats_bp_avg <- stats_bp %>% 
	group_by(encoding, gram_len, embedding_size) %>% 
	summarize(minF = min(f), maxF = max(f), avgF = mean(f), stdF = sd(f), GCI = (CI(f, ci = 0.95)[1] - CI(f, ci = 0.95)[2]))


stats_bp_avg[which.max(stats_bp_avg$avgF),]

stats_bp_avg$encoding <- as.character(stats_bp_avg$encoding)
stats_bp_avg$encoding[stats_bp_avg$encoding == 'oh'] <- 'O'
stats_bp_avg$encoding <- as.factor(stats_bp_avg$encoding)

stats_bp_avg$encoding <- as.character(stats_bp_avg$encoding)
stats_bp_avg$encoding[stats_bp_avg$encoding == 'ad'] <- 'T'
stats_bp_avg$encoding <- as.factor(stats_bp_avg$encoding)


stats_bp_avg <- stats_bp_avg %>% 
  mutate(algo = case_when(embedding_size == 0 ~ paste0(encoding, "-", gram_len),
                          TRUE ~ paste0(encoding, "-", gram_len, "-", embedding_size)))

# change factor order level for correct x-axis ordering
stats_bp_avg$algo <- factor(stats_bp_avg$algo, 
	levels=stats_bp_avg$algo[order(stats_bp_avg$encoding, stats_bp_avg$gram_len, stats_bp_avg$embedding_size, decreasing = TRUE)])




ggplot(stats_bp_avg, aes(x=algo, y=avgF, group=encoding, fill=encoding)) +
	geom_errorbar(aes(ymin=avgF-GCI, ymax=avgF+GCI), width = 0.5, size = 0.8, alpha = 0.7, color="red") +
	geom_bar(stat="identity", alpha = 0.7, show.legend = FALSE) +
	labs(title="", x ="", y = "Fmax", color="Encoding") +
	coord_flip(ylim=c(0.29,0.36)) +
	theme(text = element_text(size=20), axis.text.x = element_text(size=25), axis.text.y = element_text(size=15))
	
ggsave("bp.png", width = 7, height = 4)

# -------------------------------------------------------------------------------------------------------------



stats <- read.csv("results_chemical_15.csv")


stats_avg <- stats %>% 
	group_by(subontology) %>% 
	summarize(minF = min(f), maxF = max(f), avgF = mean(f), stdF = sd(f), GCI = (CI(f, ci = 0.95)[1] - CI(f, ci = 0.95)[2]))



# -------------------------------------------------------------------------------------------------------------

stats <- read.csv("results_tpe_15.csv")


stats_avg <- stats %>% 
	group_by(subontology) %>% 
	summarize(minF = min(f), maxF = max(f), avgF = mean(f), stdF = sd(f), GCI = (CI(f, ci = 0.95)[1] - CI(f, ci = 0.95)[2]))



# -------------------------------------------------------------------------------------------------------------


stats <- read.csv("results_bottleneck_15.csv")


stats_avg <- stats %>% 
	group_by(subontology) %>% 
	summarize(minF = min(f), maxF = max(f), avgF = mean(f), stdF = sd(f), GCI = (CI(f, ci = 0.95)[1] - CI(f, ci = 0.95)[2]))



# -------------------------------------------------------------------------------------------------------------





