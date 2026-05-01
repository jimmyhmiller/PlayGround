// Section component — renders one project. Each section gets its own bg/fg/accent
// via CSS custom properties so the page reads as a stack of distinct cards.

const ProjectSection = ({ project, index, total, density }) => {
  const {
    id, title, tagline, year, role, stack, status, repo,
    bg, fg, accent, blurb, kind, layout,
  } = project;

  const Visual = VISUAL_MAP[id];

  const renderArtifact = () => {
    if (kind === "code") {
      return <CodeBlock code={project.code} accent={accent} />;
    }
    if (kind === "code+table") {
      return (
        <div className="stack-artifact">
          <CodeBlock code={project.code} accent={accent} />
          <PerfTable table={project.table} accent={accent} />
        </div>
      );
    }
    if (kind === "video+code+table") {
      return (
        <div className="stack-artifact">
          <figure className="imgblock">
            <video src={project.video} controls loop muted playsInline preload="metadata" />
            {project.videoCaption && <figcaption>{project.videoCaption}</figcaption>}
          </figure>
          <CodeBlock code={project.code} accent={accent} />
          <PerfTable table={project.table} accent={accent} />
        </div>
      );
    }
    if (kind === "code+arch") {
      return (
        <div className="stack-artifact">
          <CodeBlock code={project.code} accent={accent} />
          <ArchList arch={project.arch} accent={accent} />
        </div>
      );
    }
    if (kind === "image") {
      return (
        <figure className="imgblock">
          <img src={project.image} alt={project.title} />
          {project.imageCaption && <figcaption>{project.imageCaption}</figcaption>}
        </figure>
      );
    }
    if (kind === "image-phone") {
      return (
        <figure className="imgblock imgblock-phone">
          <img src={project.image} alt={project.title} />
          {project.imageCaption && <figcaption>{project.imageCaption}</figcaption>}
        </figure>
      );
    }
    if (kind === "video") {
      return (
        <figure className="imgblock">
          <video src={project.video} controls loop muted playsInline preload="metadata" />
          {project.videoCaption && <figcaption>{project.videoCaption}</figcaption>}
        </figure>
      );
    }
    if (kind === "terminal") {
      return <TerminalBlock lines={project.terminal} accent={accent} label={`~/${id}`} />;
    }
    if (kind === "visual" && project.visualKind === "browser") {
      return (
        <BrowserFrame url={project.visualLabel} accent={accent}>
          {Visual ? <Visual accent={accent} /> : null}
        </BrowserFrame>
      );
    }
    if (kind === "visual" && project.visualKind === "phone") {
      return (
        <PhoneFrame accent={accent}>
          {Visual ? <Visual accent={accent} /> : null}
        </PhoneFrame>
      );
    }
    return null;
  };

  const heading = (
    <>
      <Reveal as="h2" className="project-title">{title}</Reveal>
      <Reveal as="p" className="project-tagline" delay={60}>{tagline}</Reveal>
      <Reveal as="p" className="project-blurb" delay={120}>{blurb}</Reveal>
    </>
  );

  let body;
  if (layout === "wide") {
    body = (
      <>
        <div className="project-text project-text-wide">{heading}</div>
        <Reveal className="project-artifact" delay={120}>
          {renderArtifact()}
        </Reveal>
      </>
    );
  } else if (layout === "narrow") {
    body = (
      <>
        <div className="project-text">{heading}</div>
        <Reveal className="project-artifact project-artifact-phone" delay={120}>
          {renderArtifact()}
        </Reveal>
      </>
    );
  } else if (layout === "code-left") {
    body = (
      <>
        <Reveal className="project-artifact" delay={120}>
          {renderArtifact()}
        </Reveal>
        <div className="project-text">{heading}</div>
      </>
    );
  } else {
    body = (
      <>
        <div className="project-text">{heading}</div>
        <Reveal className="project-artifact" delay={120}>
          {renderArtifact()}
        </Reveal>
      </>
    );
  }

  return (
    <section
      data-screen-label={`${String(index + 1).padStart(2, "0")} ${title}`}
      data-section-theme={project.theme || "dark"}
      className={`project project-${layout} density-${density}`}
      style={{
        "--proj-bg": bg,
        "--proj-fg": fg,
        "--accent": accent,
      }}
      id={id}
    >
      <div className="project-inner">{body}</div>
    </section>
  );
};

window.ProjectSection = ProjectSection;
