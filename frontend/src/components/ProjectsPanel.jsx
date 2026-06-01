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
      <section className="workspace-folder workspace-folder--picker">
        <header className="workspace-masthead">
          <TrackMotif className="workspace-motif" />
          <div className="workspace-masthead-row">
            <div className="workspace-masthead-copy">
              <p className="home-kicker">Workspaces</p>
              <h1 className="workspace-title">Projects</h1>
              <p className="workspace-lead">
                Choose a project, upload your files, then pick a tool to run analysis.
              </p>
            </div>
            <button className="auremin-button" onClick={onOpenCreate} type="button">
              <Plus size={15} />
              New project
            </button>
          </div>
        </header>

        <div className="workspace-assets">
          {errorMessage && <p className="page-error">{errorMessage}</p>}

          {projects.length === 0 ? (
            <div className="workspace-empty">
              <p>No projects yet</p>
              <span>Create a workspace to upload sequence files and run the HMM decoder.</span>
              <button className="auremin-button" onClick={onOpenCreate} type="button">
                Create your first project
                <ArrowRight size={16} />
              </button>
            </div>
          ) : (
            <ul className="project-list project-list--workspace">
              {projects.map((project) => (
                <li key={project.id}>
                  <button
                    className="project-row"
                    onClick={() => onOpenProject(project.id)}
                    type="button"
                  >
                    <TrackMotif className="project-row-motif" />
                    <div className="project-row-body">
                      <div className="project-row-main">
                        <strong>{project.name}</strong>
                        <span>
                          {project.runCount ?? 0} run{(project.runCount ?? 0) === 1 ? "" : "s"} ·{" "}
                          {formatDate(project.updatedAt)}
                        </span>
                      </div>
                      <ArrowRight size={18} />
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>

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
