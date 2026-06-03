import { ArrowRight, Plus } from "lucide-react";
import { motion } from "framer-motion";
import { TrackMotif } from "./TrackMotif.jsx";
import { premiumEase } from "../lib/motion.js";

function formatDate(value) {
  return new Date(value).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function ProjectsPanel({
  creatingProject,
  errorMessage,
  newProjectName,
  onCreateProject,
  onOpenCreate,
  onOpenProject,
  onSetCreatingProject,
  onSetNewProjectName,
  projects,
}) {
  return (
    <>
      <main className="projects-page">
        <div className="projects-page-layout">
          <section className="projects-page-main">
            <header className="projects-page-head">
              <TrackMotif className="projects-page-motif" />
              <div className="projects-page-title-row">
                <div className="projects-page-copy">
                  <h1>Projects</h1>
                  <p className="projects-page-lead">
                    Open a workspace to organize sequence data, run analyses, and review results.
                  </p>
                </div>
                <button className="auremin-button" onClick={onOpenCreate} type="button">
                  <Plus size={15} />
                  New project
                </button>
              </div>
              {projects.length > 0 && (
                <p className="projects-page-count">
                  {projects.length} workspace{projects.length === 1 ? "" : "s"}
                </p>
              )}
            </header>

            <div className="projects-page-body">
              {errorMessage && <p className="page-error">{errorMessage}</p>}

              {projects.length === 0 ? (
                <div className="projects-empty">
                  <p>No projects yet</p>
                  <span>Create a workspace to upload genomes and start your first analysis.</span>
                  <button className="auremin-button" onClick={onOpenCreate} type="button">
                    Create your first project
                    <ArrowRight size={16} />
                  </button>
                </div>
              ) : (
                <ul className="projects-list">
                  {projects.map((project) => (
                    <li key={project.id}>
                      <button
                        className="projects-card"
                        onClick={() => onOpenProject(project.id)}
                        type="button"
                      >
                        <div className="projects-card-main">
                          <strong>{project.name}</strong>
                          <span>
                            {project.runCount ?? 0} run{(project.runCount ?? 0) === 1 ? "" : "s"} ·{" "}
                            {formatDate(project.updatedAt)}
                          </span>
                        </div>
                        <ArrowRight size={18} />
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>

          <aside className="projects-page-aside" aria-label="Workspace overview">
            <p className="home-kicker">Genome research IDE</p>
            <h2 className="projects-aside-title">Work locally, end to end</h2>
            <p className="projects-aside-lead">
              Each project is a self-contained workspace on your machine — sequence files, analysis
              runs, and interactive results stay together without leaving the app.
            </p>

            <ol className="projects-aside-steps">
              <li>
                <span className="projects-aside-step-label">Inputs</span>
                <span>Upload and organize FASTA and related files.</span>
              </li>
              <li>
                <span className="projects-aside-step-label">Tools</span>
                <span>Run gene annotation and other analyses on your data.</span>
              </li>
              <li>
                <span className="projects-aside-step-label">Runs</span>
                <span>Explore tracks, gene lists, and exportable results.</span>
              </li>
            </ol>

            <dl className="projects-aside-specs">
              <div>
                <dt>Storage</dt>
                <dd>Local only</dd>
              </div>
              <div>
                <dt>Formats</dt>
                <dd>FASTA, GFF3, BED</dd>
              </div>
            </dl>

            <a
              className="projects-aside-link"
              href="https://github.com/sansen405/genome-annotation-hidden-markov-model"
              rel="noopener noreferrer"
              target="_blank"
            >
              Read the docs
              <ArrowRight size={15} />
            </a>
          </aside>
        </div>
      </main>

      {creatingProject && (
        <div className="modal-backdrop">
          <motion.form
            animate={{ opacity: 1, scale: 1, y: 0 }}
            className="modal-card card"
            initial={{ opacity: 0, scale: 0.98, y: 12 }}
            onSubmit={onCreateProject}
            transition={{ duration: 0.5, ease: premiumEase }}
          >
            <h2>New project</h2>
            <p className="modal-copy">
              Name your workspace — you can upload FASTA files after opening it.
            </p>
            <label className="field-label">
              Name
              <input
                autoFocus
                onChange={(event) => onSetNewProjectName(event.target.value)}
                placeholder="e.g. S. pombe chromosome III"
                value={newProjectName}
              />
            </label>
            <div className="modal-actions">
              <button onClick={() => onSetCreatingProject(false)} type="button">
                Cancel
              </button>
              <button
                className="auremin-button auremin-button--solid"
                disabled={!newProjectName.trim()}
                type="submit"
              >
                Create
              </button>
            </div>
          </motion.form>
        </div>
      )}
    </>
  );
}
