import { ArrowRight } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import { DnaStrand } from "./DnaStrand.jsx";
import { fadeIn, lineReveal, premiumEase, staggerContainer } from "../lib/motion.js";

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
  const heroLines = ["Decode genomes", "on your machine."];

  return (
    <main className="landing-page">
      <div className="landing-backdrop" aria-hidden="true">
        <DnaStrand className="landing-backdrop-helix" />
      </div>

      <div className="landing-content">
        <motion.div
          animate={reduceMotion ? undefined : "visible"}
          className="landing-helix-mark"
          initial={reduceMotion ? undefined : "hidden"}
          variants={fadeIn}
        >
          <DnaStrand className="landing-helix-icon" />
        </motion.div>

        <motion.p
          animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
          className="landing-kicker"
          initial={reduceMotion ? undefined : { opacity: 0, y: 16 }}
          transition={{ delay: 0.1, duration: 0.8, ease: premiumEase }}
        >
          Gene annotation
        </motion.p>

        <HeroTitle lines={heroLines} />

        <motion.p
          animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
          className="landing-lead"
          initial={reduceMotion ? undefined : { opacity: 0, y: 20 }}
          transition={{ delay: 0.28, duration: 0.85, ease: premiumEase }}
        >
          Upload FASTA, run the HMM decoder with CNN splice scores, and explore genes on an
          interactive track — all locally, with no cloud upload.
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
              {projectCount} workspace{projectCount === 1 ? "" : "s"} saved on this machine
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
            <dt>Model</dt>
            <dd>21-state HMM</dd>
          </div>
          <div>
            <dt>Splice sites</dt>
            <dd>Calibrated CNN</dd>
          </div>
          <div>
            <dt>Export</dt>
            <dd>GFF3, CSV, BED</dd>
          </div>
        </motion.dl>
      </div>
    </main>
  );
}
