export const premiumEase = [0.76, 0, 0.24, 1];

export const revealTransition = {
  duration: 0.95,
  ease: premiumEase,
};

export const staggerContainer = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.11,
      delayChildren: 0.06,
    },
  },
};

export const fadeUp = {
  hidden: {
    opacity: 0,
    y: 36,
  },
  visible: {
    opacity: 1,
    y: 0,
    transition: revealTransition,
  },
};

export const fadeIn = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { duration: 0.8, ease: premiumEase },
  },
};

export const lineReveal = {
  hidden: {
    opacity: 0,
    y: "110%",
  },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 1,
      ease: premiumEase,
    },
  },
};
