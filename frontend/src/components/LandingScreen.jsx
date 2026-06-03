import { ArrowRight } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import { DnaStrand } from "./DnaStrand.jsx";
import { lineReveal, premiumEase, staggerContainer } from "../lib/motion.js";

function HeroTitle({ lines }) {
  const reduceMotion = useReducedMotion();

  if (reduceMotion) {
    return (
      <h1 className="landing-title">
        {lines.map((line) => (
          <span className="landing-title-line" key={line}>
            {line}
          </span>
        ))}
      </h1>
    );
  }

  return (
    <motion.h1
      className="landing-title"
      initial="hidden"
      animate="visible"
      variants={staggerContainer}
    >
      {lines.map((line) => (
        <span className="landing-title-line-mask" key={line}>
          <motion.span className="landing-title-line" variants={lineReveal}>
            {line}
          </motion.span>
        </span>
      ))}
    </motion.h1>
  );
}

export function LandingScreen({ onEnter, projectCount = 0 }) {
  const reduceMotion = useReducedMotion();
  const heroLines = ["Built for", "genome research"];

  return (
    <main className="landing-page">
      <div className="landing-backdrop" aria-hidden="true">
        <DnaStrand className="landing-backdrop-helix" />
      </div>

      <div className="landing-content">
        <HeroTitle lines={heroLines} />

        <motion.p
          animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
          className="landing-lead"
          initial={reduceMotion ? undefined : { opacity: 0, y: 20 }}
          transition={{ delay: 0.28, duration: 0.85, ease: premiumEase }}
        >
          Organize sequence data, run analyses, and explore results in one local workspace —
          built for end-to-end genome research, with no cloud upload.
        </motion.p>

        <motion.div
          animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
          className="landing-actions"
          initial={reduceMotion ? undefined : { opacity: 0, y: 20 }}
          transition={{ delay: 0.4, duration: 0.85, ease: premiumEase }}
        >
          <button className="auremin-button auremin-button--solid landing-enter" onClick={onEnter} type="button">
            Get started
            <ArrowRight size={17} />
          </button>
          {projectCount > 0 && (
            <p className="landing-resume">
              {projectCount} project{projectCount === 1 ? "" : "s"} saved on this machine
            </p>
          )}
        </motion.div>

        <motion.dl
          animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
          className="landing-specs"
          initial={reduceMotion ? undefined : { opacity: 0, y: 20 }}
          transition={{ delay: 0.52, duration: 0.85, ease: premiumEase }}
        >
          <div>
            <dt>Analyze</dt>
            <dd>Pipelines & tools</dd>
          </div>
          <div>
            <dt>Research</dt>
            <dd>Tracks & exploration</dd>
          </div>
          <div>
            <dt>Model</dt>
            <dd>HMM + CNN</dd>
          </div>
        </motion.dl>
      </div>
    </main>
  );
}
