import { ArrowRight, Plus } from "lucide-react";
import { motion } from "framer-motion";
import { DnaStrand } from "./DnaStrand.jsx";
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
        <header className="projects-page-head">
          <DnaStrand className="projects-page-helix" />
          <p className="home-kicker">Workspaces</p>
          <div className="projects-page-title-row">
            <div className="projects-page-copy">
              <h1>Projects</h1>
              <p className="projects-page-lead">
                Choose a project, upload your files, then pick a tool to run analysis.
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
              <span>Create a workspace to upload sequence files and run gene annotation.</span>
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
