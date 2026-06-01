import Lenis from "lenis";
import { useEffect } from "react";

export function useHomeLenis(onScrollChange) {
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.15,
      easing: (t) => Math.min(1, 1.001 - 2 ** (-10 * t)),
      smoothWheel: true,
      touchMultiplier: 1.1,
    });

    document.documentElement.classList.add("lenis", "lenis-smooth");

    lenis.on("scroll", ({ scroll }) => {
      onScrollChange(scroll);
    });

    let frame = 0;
    function raf(time) {
      lenis.raf(time);
      frame = requestAnimationFrame(raf);
    }
    frame = requestAnimationFrame(raf);

    return () => {
      cancelAnimationFrame(frame);
      document.documentElement.classList.remove("lenis", "lenis-smooth");
      lenis.destroy();
      onScrollChange(0);
      window.scrollTo(0, 0);
    };
  }, [onScrollChange]);
}
