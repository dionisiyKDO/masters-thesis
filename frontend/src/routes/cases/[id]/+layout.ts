import type { LayoutLoad } from "./$types";

export const load: LayoutLoad = ({ params }) => {
    return { caseId: params.id };
};