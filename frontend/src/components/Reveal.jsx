import { motion, useReducedMotion } from "framer-motion";
import { fadeUp } from "../lib/motion.js";

export function Reveal({
  as = "div",
  children,
  className = "",
  delay = 0,
  once = true,
  viewport = { amount: 0.2, margin: "0px 0px -8% 0px" },
  ...props
}) {
  const reduceMotion = useReducedMotion();
  const Component = motion[as] ?? motion.div;

  if (reduceMotion) {
    const Static = as;
    return (
      <Static className={className} {...props}>
        {children}
      </Static>
    );
  }

  return (
    <Component
      className={className}
      initial="hidden"
      variants={{
        hidden: fadeUp.hidden,
        visible: {
          ...fadeUp.visible,
          transition: {
            ...fadeUp.visible.transition,
            delay,
          },
        },
      }}
      viewport={once ? { ...viewport, once: true } : viewport}
      whileInView="visible"
      {...props}
    >
      {children}
    </Component>
  );
}
