
const api_prefix = "api"
const pdf_prefix = "pdfs"
const requirement_prefix = "requirements"
const bundle_prefix = "bundles"

export const routes = {
    upload_pdf: `/${api_prefix}/${pdf_prefix}/upload`,
    get_all_pdfs: `/${api_prefix}/${pdf_prefix}`,
    download_pdf: (pdf_id: number) => `/${api_prefix}/${pdf_prefix}/${pdf_id}/download`,
    get_requirement: (requirement_id: number) => `/${api_prefix}/${requirement_prefix}/${requirement_id}`,
    get_bundle: (bundle_id: number) => `/${api_prefix}/${bundle_prefix}/${bundle_id}`,
    get_all_bundles: `/${api_prefix}/${bundle_prefix}`,
    delete_pdf: (pdf_id: number) => `/${api_prefix}/${pdf_prefix}/${pdf_id}/delete`,
    save_requirement: `/${api_prefix}/${requirement_prefix}`,
    generate_bundle: `/${api_prefix}/${bundle_prefix}/generate`,
}